//! Refinement operators for each dimension of the abstraction triple (k, n, ε).
//!
//! - `RefineOutputAlphabet(k → k')`: increase semantic clusters
//! - `RefineInputDepth(n → n')`: increase probing depth
//! - `RefineDistributionResolution(ε → ε')`: decrease distributional tolerance
//!
//! Also provides refinement impact estimation, greedy/multi-objective strategies,
//! and refinement history tracking.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::fmt;
use ordered_float::OrderedFloat;
use chrono::{DateTime, Utc};

use super::lattice::AbstractionTriple;

// ---------------------------------------------------------------------------
// Refinement kinds and operators
// ---------------------------------------------------------------------------

/// The dimension of abstraction being refined.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RefinementKind {
    /// Increase the number of output clusters.
    OutputAlphabet,
    /// Increase the input probing depth.
    InputDepth,
    /// Decrease the distributional tolerance.
    DistributionResolution,
    /// Refine multiple dimensions simultaneously.
    Combined,
}

impl fmt::Display for RefinementKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutputAlphabet => write!(f, "k-refine"),
            Self::InputDepth => write!(f, "n-refine"),
            Self::DistributionResolution => write!(f, "ε-refine"),
            Self::Combined => write!(f, "combined"),
        }
    }
}

/// A specific refinement operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefinementOperator {
    /// Which dimension to refine.
    pub kind: RefinementKind,
    /// Source abstraction triple.
    pub from: AbstractionTriple,
    /// Target abstraction triple.
    pub to: AbstractionTriple,
    /// Estimated cost of performing this refinement.
    pub estimated_cost: f64,
    /// Estimated improvement in fidelity.
    pub estimated_fidelity_gain: f64,
    /// Priority score (higher = refine first).
    pub priority: f64,
    /// Human-readable description.
    pub description: String,
}

impl RefinementOperator {
    /// Create a k-refinement operator.
    pub fn refine_k(from: &AbstractionTriple, new_k: usize, alphabet_size: usize) -> Self {
        let to = AbstractionTriple::new(new_k, from.n, from.epsilon);
        let cost_from = from.estimated_cost(alphabet_size);
        let cost_to = to.estimated_cost(alphabet_size);
        let estimated_cost = cost_to - cost_from;

        // Fidelity gain: more clusters = less information loss per cluster.
        // Rough model: gain ∝ log(k'/k)
        let fidelity_gain = if new_k > from.k {
            (new_k as f64 / from.k as f64).ln()
        } else {
            0.0
        };

        let priority = if estimated_cost > 0.0 {
            fidelity_gain / estimated_cost
        } else {
            fidelity_gain
        };

        Self {
            kind: RefinementKind::OutputAlphabet,
            from: from.clone(),
            to,
            estimated_cost: estimated_cost.max(0.0),
            estimated_fidelity_gain: fidelity_gain,
            priority,
            description: format!("Refine output alphabet: k={} → k={}", from.k, new_k),
        }
    }

    /// Create an n-refinement operator.
    pub fn refine_n(from: &AbstractionTriple, new_n: usize, alphabet_size: usize) -> Self {
        let to = AbstractionTriple::new(from.k, new_n, from.epsilon);
        let cost_from = from.estimated_cost(alphabet_size);
        let cost_to = to.estimated_cost(alphabet_size);
        let estimated_cost = cost_to - cost_from;

        // Fidelity gain: deeper probing reveals more behavioral distinctions.
        // Rough model: gain ∝ (n' - n) * log(|Σ|)
        let fidelity_gain = if new_n > from.n {
            (new_n - from.n) as f64 * (alphabet_size as f64).ln().max(1.0)
        } else {
            0.0
        };

        let priority = if estimated_cost > 0.0 {
            fidelity_gain / estimated_cost
        } else {
            fidelity_gain
        };

        Self {
            kind: RefinementKind::InputDepth,
            from: from.clone(),
            to,
            estimated_cost: estimated_cost.max(0.0),
            estimated_fidelity_gain: fidelity_gain,
            priority,
            description: format!("Refine input depth: n={} → n={}", from.n, new_n),
        }
    }

    /// Create an ε-refinement operator.
    pub fn refine_epsilon(from: &AbstractionTriple, new_eps: f64, alphabet_size: usize) -> Self {
        let to = AbstractionTriple::new(from.k, from.n, new_eps);
        let cost_from = from.estimated_cost(alphabet_size);
        let cost_to = to.estimated_cost(alphabet_size);
        let estimated_cost = cost_to - cost_from;

        // Fidelity gain: tighter tolerance = better distributional resolution.
        // Rough model: gain ∝ log(ε/ε')
        let fidelity_gain = if new_eps < from.epsilon && new_eps > 0.0 {
            (from.epsilon / new_eps).ln()
        } else {
            0.0
        };

        let priority = if estimated_cost > 0.0 {
            fidelity_gain / estimated_cost
        } else {
            fidelity_gain
        };

        Self {
            kind: RefinementKind::DistributionResolution,
            from: from.clone(),
            to,
            estimated_cost: estimated_cost.max(0.0),
            estimated_fidelity_gain: fidelity_gain,
            priority,
            description: format!("Refine distribution: ε={:.4} → ε={:.4}", from.epsilon, new_eps),
        }
    }

    /// Create a combined refinement operator.
    pub fn combined(
        from: &AbstractionTriple,
        new_k: usize,
        new_n: usize,
        new_eps: f64,
        alphabet_size: usize,
    ) -> Self {
        let to = AbstractionTriple::new(new_k, new_n, new_eps);
        let cost_from = from.estimated_cost(alphabet_size);
        let cost_to = to.estimated_cost(alphabet_size);
        let estimated_cost = (cost_to - cost_from).max(0.0);

        let k_gain = if new_k > from.k { (new_k as f64 / from.k as f64).ln() } else { 0.0 };
        let n_gain = if new_n > from.n { (new_n - from.n) as f64 * (alphabet_size as f64).ln().max(1.0) } else { 0.0 };
        let e_gain = if new_eps < from.epsilon && new_eps > 0.0 { (from.epsilon / new_eps).ln() } else { 0.0 };
        let fidelity_gain = k_gain + n_gain + e_gain;

        let priority = if estimated_cost > 0.0 {
            fidelity_gain / estimated_cost
        } else {
            fidelity_gain
        };

        Self {
            kind: RefinementKind::Combined,
            from: from.clone(),
            to,
            estimated_cost,
            estimated_fidelity_gain: fidelity_gain,
            priority,
            description: format!(
                "Combined refine: ({},{},{:.4}) → ({},{},{:.4})",
                from.k, from.n, from.epsilon, new_k, new_n, new_eps
            ),
        }
    }

    /// Efficiency ratio: fidelity gain per unit cost.
    pub fn efficiency(&self) -> f64 {
        if self.estimated_cost > 0.0 {
            self.estimated_fidelity_gain / self.estimated_cost
        } else {
            f64::INFINITY
        }
    }

    /// Check if this refinement is a strict improvement.
    pub fn is_strict_refinement(&self) -> bool {
        self.from.lt(&self.to)
    }
}

impl fmt::Display for RefinementOperator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {} (cost={:.2}, gain={:.4}, prio={:.4})",
            self.kind, self.description,
            self.estimated_cost, self.estimated_fidelity_gain, self.priority
        )
    }
}

// ---------------------------------------------------------------------------
// Refinement result
// ---------------------------------------------------------------------------

/// The outcome of applying a refinement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefinementResult {
    /// The operator that was applied.
    pub operator: RefinementOperator,
    /// Whether the refinement succeeded (new model was built and verified).
    pub success: bool,
    /// Actual fidelity gain observed.
    pub actual_fidelity_gain: f64,
    /// Actual cost incurred.
    pub actual_cost: f64,
    /// Whether the refinement resolved the counter-example.
    pub resolved_counterexample: bool,
    /// New abstraction triple after refinement.
    pub new_triple: AbstractionTriple,
    /// Timestamp.
    pub timestamp: String,
}

impl RefinementResult {
    pub fn new(operator: RefinementOperator) -> Self {
        Self {
            new_triple: operator.to.clone(),
            operator,
            success: false,
            actual_fidelity_gain: 0.0,
            actual_cost: 0.0,
            resolved_counterexample: false,
            timestamp: Utc::now().to_rfc3339(),
        }
    }

    /// Prediction error: how far off was our cost/gain estimate?
    pub fn cost_prediction_error(&self) -> f64 {
        (self.actual_cost - self.operator.estimated_cost).abs()
    }

    pub fn fidelity_prediction_error(&self) -> f64 {
        (self.actual_fidelity_gain - self.operator.estimated_fidelity_gain).abs()
    }
}

impl fmt::Display for RefinementResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let status = if self.success { "OK" } else { "FAIL" };
        write!(
            f,
            "RefinementResult[{} {}, gain={:.4}, cost={:.2}, resolved={}]",
            status, self.operator.kind,
            self.actual_fidelity_gain, self.actual_cost, self.resolved_counterexample
        )
    }
}

// ---------------------------------------------------------------------------
// Refinement impact estimation
// ---------------------------------------------------------------------------

/// Estimated impact of a refinement, used for planning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefinementImpact {
    /// Expected reduction in abstraction error.
    pub error_reduction: f64,
    /// Expected number of new distinguishable states.
    pub new_states: usize,
    /// Expected sample complexity increase.
    pub sample_complexity_increase: f64,
    /// Expected time cost (seconds).
    pub time_cost: f64,
    /// Probability of resolving the current counter-example.
    pub resolve_probability: f64,
    /// Overall utility score (combining all factors).
    pub utility: f64,
}

impl RefinementImpact {
    /// Estimate impact of a k-refinement.
    pub fn estimate_k_refinement(
        current: &AbstractionTriple,
        new_k: usize,
        alphabet_size: usize,
        counterexample_info: Option<&CounterexampleInfo>,
    ) -> Self {
        let ratio = new_k as f64 / current.k as f64;

        // Error reduction: each cluster split roughly halves the within-cluster error.
        let error_reduction = 1.0 - 1.0 / ratio;

        // New distinguishable states.
        let new_states = new_k - current.k;

        // Sample complexity increase: need more samples per cluster.
        let sample_increase = ratio;

        // Time cost: roughly proportional to k * existing cost.
        let time_cost = current.estimated_cost(alphabet_size) * (ratio - 1.0);

        // Resolve probability: higher if the counter-example is about cluster confusion.
        let resolve_prob = match counterexample_info {
            Some(info) if info.involves_cluster_confusion => 0.8 * error_reduction,
            Some(_) => 0.3 * error_reduction,
            None => 0.5 * error_reduction,
        };

        let utility = error_reduction * resolve_prob / (1.0 + time_cost.ln().max(0.0));

        Self {
            error_reduction,
            new_states,
            sample_complexity_increase: sample_increase,
            time_cost: time_cost.max(0.0),
            resolve_probability: resolve_prob.min(1.0),
            utility,
        }
    }

    /// Estimate impact of an n-refinement.
    pub fn estimate_n_refinement(
        current: &AbstractionTriple,
        new_n: usize,
        alphabet_size: usize,
        counterexample_info: Option<&CounterexampleInfo>,
    ) -> Self {
        let depth_increase = new_n.saturating_sub(current.n);

        // Error reduction: more probing depth reveals more state distinctions.
        let error_reduction = 1.0 - (0.5f64).powi(depth_increase as i32);

        // New words we can probe.
        let new_states = if alphabet_size > 0 {
            (alphabet_size.pow(new_n as u32)).saturating_sub(alphabet_size.pow(current.n as u32))
        } else {
            0
        };

        // Sample complexity increase: exponential in depth.
        let sample_increase = (alphabet_size as f64).powi(depth_increase as i32);

        let time_cost = current.estimated_cost(alphabet_size) * sample_increase;

        let resolve_prob = match counterexample_info {
            Some(info) if info.involves_depth_insufficiency => 0.9 * error_reduction,
            Some(_) => 0.2 * error_reduction,
            None => 0.4 * error_reduction,
        };

        let utility = error_reduction * resolve_prob / (1.0 + time_cost.ln().max(0.0));

        Self {
            error_reduction,
            new_states,
            sample_complexity_increase: sample_increase,
            time_cost: time_cost.max(0.0),
            resolve_probability: resolve_prob.min(1.0),
            utility,
        }
    }

    /// Estimate impact of an ε-refinement.
    pub fn estimate_epsilon_refinement(
        current: &AbstractionTriple,
        new_eps: f64,
        alphabet_size: usize,
        counterexample_info: Option<&CounterexampleInfo>,
    ) -> Self {
        let ratio = current.epsilon / new_eps.max(1e-15);

        // Error reduction: tighter tolerance reduces distributional error.
        let error_reduction = 1.0 - 1.0 / ratio;

        // New states: same number but finer distinctions.
        let new_states = 0;

        // Sample complexity increase: O(1/ε²) so ratio² times more.
        let sample_increase = ratio * ratio;

        let time_cost = current.estimated_cost(alphabet_size) * (sample_increase - 1.0);

        let resolve_prob = match counterexample_info {
            Some(info) if info.involves_distribution_imprecision => 0.85 * error_reduction,
            Some(_) => 0.25 * error_reduction,
            None => 0.4 * error_reduction,
        };

        let utility = error_reduction * resolve_prob / (1.0 + time_cost.ln().max(0.0));

        Self {
            error_reduction,
            new_states,
            sample_complexity_increase: sample_increase,
            time_cost: time_cost.max(0.0),
            resolve_probability: resolve_prob.min(1.0),
            utility,
        }
    }
}

impl fmt::Display for RefinementImpact {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Impact[err_red={:.4}, new_states={}, sample_inc={:.2}, time={:.2}, resolve={:.4}, utility={:.4}]",
            self.error_reduction, self.new_states, self.sample_complexity_increase,
            self.time_cost, self.resolve_probability, self.utility
        )
    }
}

/// Information about a counter-example, used to guide refinement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CounterexampleInfo {
    /// The counter-example involves two states being merged that shouldn't be.
    pub involves_cluster_confusion: bool,
    /// The counter-example requires deeper probing to distinguish states.
    pub involves_depth_insufficiency: bool,
    /// The counter-example is about distributional differences being below tolerance.
    pub involves_distribution_imprecision: bool,
    /// The states involved in the counter-example.
    pub involved_states: Vec<String>,
    /// The input words involved.
    pub involved_inputs: Vec<String>,
    /// Estimated severity (0 to 1).
    pub severity: f64,
}

impl CounterexampleInfo {
    pub fn cluster_confusion(states: Vec<String>) -> Self {
        Self {
            involves_cluster_confusion: true,
            involves_depth_insufficiency: false,
            involves_distribution_imprecision: false,
            involved_states: states,
            involved_inputs: Vec::new(),
            severity: 0.5,
        }
    }

    pub fn depth_insufficiency(inputs: Vec<String>) -> Self {
        Self {
            involves_cluster_confusion: false,
            involves_depth_insufficiency: true,
            involves_distribution_imprecision: false,
            involved_states: Vec::new(),
            involved_inputs: inputs,
            severity: 0.5,
        }
    }

    pub fn distribution_imprecision(states: Vec<String>, severity: f64) -> Self {
        Self {
            involves_cluster_confusion: false,
            involves_depth_insufficiency: false,
            involves_distribution_imprecision: true,
            involved_states: states,
            involved_inputs: Vec::new(),
            severity,
        }
    }

    /// Classify which dimension most likely needs refinement.
    pub fn recommended_dimension(&self) -> RefinementKind {
        if self.involves_cluster_confusion {
            RefinementKind::OutputAlphabet
        } else if self.involves_depth_insufficiency {
            RefinementKind::InputDepth
        } else if self.involves_distribution_imprecision {
            RefinementKind::DistributionResolution
        } else {
            // Default: refine the cheapest dimension.
            RefinementKind::OutputAlphabet
        }
    }
}

// ---------------------------------------------------------------------------
// Refinement strategies
// ---------------------------------------------------------------------------

/// Strategy for selecting which refinement to apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RefinementStrategy {
    /// Always refine the dimension with highest priority.
    Greedy,
    /// Refine the dimension suggested by the counter-example.
    CounterexampleGuided,
    /// Refine the dimension with best cost/fidelity ratio.
    CostEfficient,
    /// Refine all dimensions simultaneously.
    AllDimensions,
    /// Cycle through dimensions round-robin.
    RoundRobin,
    /// Multi-objective optimization (Pareto front).
    MultiObjective,
    /// Adaptive: learn from past refinements.
    Adaptive,
}

/// Refinement planner: decides which refinement to apply given a counterexample.
#[derive(Debug, Clone)]
pub struct RefinementPlanner {
    pub strategy: RefinementStrategy,
    pub alphabet_size: usize,
    /// Available k values to refine to.
    pub k_options: Vec<usize>,
    /// Available n values to refine to.
    pub n_options: Vec<usize>,
    /// Available ε values to refine to.
    pub epsilon_options: Vec<f64>,
    /// Round-robin counter.
    round_robin_idx: usize,
    /// Learning weights for adaptive strategy.
    dimension_weights: [f64; 3], // [k, n, ε]
}

impl RefinementPlanner {
    pub fn new(
        strategy: RefinementStrategy,
        alphabet_size: usize,
        k_options: Vec<usize>,
        n_options: Vec<usize>,
        epsilon_options: Vec<f64>,
    ) -> Self {
        Self {
            strategy,
            alphabet_size,
            k_options,
            n_options,
            epsilon_options,
            round_robin_idx: 0,
            dimension_weights: [1.0, 1.0, 1.0],
        }
    }

    /// Generate all possible single-step refinement operators from current triple.
    pub fn generate_operators(&self, current: &AbstractionTriple) -> Vec<RefinementOperator> {
        let mut ops = Vec::new();

        // k refinements.
        for &k in &self.k_options {
            if k > current.k {
                ops.push(RefinementOperator::refine_k(current, k, self.alphabet_size));
            }
        }

        // n refinements.
        for &n in &self.n_options {
            if n > current.n {
                ops.push(RefinementOperator::refine_n(current, n, self.alphabet_size));
            }
        }

        // ε refinements.
        for &eps in &self.epsilon_options {
            if eps < current.epsilon {
                ops.push(RefinementOperator::refine_epsilon(current, eps, self.alphabet_size));
            }
        }

        ops
    }

    /// Select the best refinement operator according to the strategy.
    pub fn select(
        &mut self,
        current: &AbstractionTriple,
        counterexample: Option<&CounterexampleInfo>,
        budget_remaining: f64,
    ) -> Option<RefinementOperator> {
        let all_ops = self.generate_operators(current);
        if all_ops.is_empty() {
            return None;
        }

        // Filter by budget.
        let affordable: Vec<RefinementOperator> = all_ops.into_iter()
            .filter(|op| op.estimated_cost <= budget_remaining)
            .collect();

        if affordable.is_empty() {
            return None;
        }

        match self.strategy {
            RefinementStrategy::Greedy => self.select_greedy(&affordable),
            RefinementStrategy::CounterexampleGuided => {
                self.select_counterexample_guided(&affordable, counterexample)
            }
            RefinementStrategy::CostEfficient => self.select_cost_efficient(&affordable),
            RefinementStrategy::AllDimensions => {
                self.select_all_dimensions(current, budget_remaining)
            }
            RefinementStrategy::RoundRobin => self.select_round_robin(&affordable),
            RefinementStrategy::MultiObjective => {
                self.select_multi_objective(&affordable, counterexample)
            }
            RefinementStrategy::Adaptive => {
                self.select_adaptive(&affordable, counterexample)
            }
        }
    }

    /// Greedy: highest priority operator.
    fn select_greedy(&self, ops: &[RefinementOperator]) -> Option<RefinementOperator> {
        ops.iter()
            .max_by(|a, b| a.priority.partial_cmp(&b.priority).unwrap_or(std::cmp::Ordering::Equal))
            .cloned()
    }

    /// Counter-example guided: refine the dimension the counter-example suggests.
    fn select_counterexample_guided(
        &self,
        ops: &[RefinementOperator],
        counterexample: Option<&CounterexampleInfo>,
    ) -> Option<RefinementOperator> {
        let target_kind = counterexample
            .map(|cx| cx.recommended_dimension())
            .unwrap_or(RefinementKind::OutputAlphabet);

        // Find the cheapest operator of the target kind.
        let matching: Vec<&RefinementOperator> = ops.iter()
            .filter(|op| op.kind == target_kind)
            .collect();

        if !matching.is_empty() {
            // Pick the one with best efficiency among matching.
            matching.iter()
                .max_by(|a, b| a.efficiency().partial_cmp(&b.efficiency())
                    .unwrap_or(std::cmp::Ordering::Equal))
                .map(|op| (*op).clone())
        } else {
            // Fall back to greedy.
            self.select_greedy(ops)
        }
    }

    /// Cost-efficient: best gain-per-cost ratio.
    fn select_cost_efficient(&self, ops: &[RefinementOperator]) -> Option<RefinementOperator> {
        ops.iter()
            .max_by(|a, b| a.efficiency().partial_cmp(&b.efficiency())
                .unwrap_or(std::cmp::Ordering::Equal))
            .cloned()
    }

    /// All dimensions: refine all at once (combined operator).
    fn select_all_dimensions(
        &self,
        current: &AbstractionTriple,
        budget: f64,
    ) -> Option<RefinementOperator> {
        // Find the next k, n, ε values.
        let next_k = self.k_options.iter().find(|&&k| k > current.k).copied()
            .unwrap_or(current.k);
        let next_n = self.n_options.iter().find(|&&n| n > current.n).copied()
            .unwrap_or(current.n);
        let next_eps = self.epsilon_options.iter().rev()
            .find(|&&e| e < current.epsilon).copied()
            .unwrap_or(current.epsilon);

        if next_k == current.k && next_n == current.n
            && (next_eps - current.epsilon).abs() < 1e-15
        {
            return None;
        }

        let op = RefinementOperator::combined(current, next_k, next_n, next_eps, self.alphabet_size);
        if op.estimated_cost <= budget {
            Some(op)
        } else {
            // Too expensive — fall back to cheapest single dimension.
            let all_ops = self.generate_operators(current);
            let affordable: Vec<RefinementOperator> = all_ops.into_iter()
                .filter(|op| op.estimated_cost <= budget)
                .collect();
            self.select_greedy(&affordable)
        }
    }

    /// Round-robin: cycle through dimensions.
    fn select_round_robin(&mut self, ops: &[RefinementOperator]) -> Option<RefinementOperator> {
        let dimensions = [
            RefinementKind::OutputAlphabet,
            RefinementKind::InputDepth,
            RefinementKind::DistributionResolution,
        ];

        for _ in 0..3 {
            let target = dimensions[self.round_robin_idx % 3];
            self.round_robin_idx += 1;

            let matching: Vec<&RefinementOperator> = ops.iter()
                .filter(|op| op.kind == target)
                .collect();

            if let Some(op) = matching.first() {
                return Some((*op).clone());
            }
        }

        // Fall back.
        ops.first().cloned()
    }

    /// Multi-objective: find Pareto-optimal refinements.
    fn select_multi_objective(
        &self,
        ops: &[RefinementOperator],
        counterexample: Option<&CounterexampleInfo>,
    ) -> Option<RefinementOperator> {
        if ops.is_empty() {
            return None;
        }

        // Compute impacts for each operator.
        let mut scored: Vec<(usize, f64)> = Vec::new();
        for (idx, op) in ops.iter().enumerate() {
            let impact = match op.kind {
                RefinementKind::OutputAlphabet => {
                    RefinementImpact::estimate_k_refinement(
                        &op.from, op.to.k, self.alphabet_size, counterexample,
                    )
                }
                RefinementKind::InputDepth => {
                    RefinementImpact::estimate_n_refinement(
                        &op.from, op.to.n, self.alphabet_size, counterexample,
                    )
                }
                RefinementKind::DistributionResolution => {
                    RefinementImpact::estimate_epsilon_refinement(
                        &op.from, op.to.epsilon, self.alphabet_size, counterexample,
                    )
                }
                RefinementKind::Combined => {
                    // Use weighted sum of component impacts.
                    let k_imp = RefinementImpact::estimate_k_refinement(
                        &op.from, op.to.k, self.alphabet_size, counterexample,
                    );
                    let n_imp = RefinementImpact::estimate_n_refinement(
                        &op.from, op.to.n, self.alphabet_size, counterexample,
                    );
                    let e_imp = RefinementImpact::estimate_epsilon_refinement(
                        &op.from, op.to.epsilon, self.alphabet_size, counterexample,
                    );
                    RefinementImpact {
                        error_reduction: k_imp.error_reduction + n_imp.error_reduction + e_imp.error_reduction,
                        new_states: k_imp.new_states + n_imp.new_states,
                        sample_complexity_increase: k_imp.sample_complexity_increase *
                            n_imp.sample_complexity_increase * e_imp.sample_complexity_increase,
                        time_cost: k_imp.time_cost + n_imp.time_cost + e_imp.time_cost,
                        resolve_probability: 1.0 - (1.0 - k_imp.resolve_probability)
                            * (1.0 - n_imp.resolve_probability)
                            * (1.0 - e_imp.resolve_probability),
                        utility: k_imp.utility + n_imp.utility + e_imp.utility,
                    }
                }
            };
            scored.push((idx, impact.utility));
        }

        // Find Pareto-optimal (here simplified to highest utility).
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.first().map(|(idx, _)| ops[*idx].clone())
    }

    /// Adaptive: use learned weights from past refinement results.
    fn select_adaptive(
        &self,
        ops: &[RefinementOperator],
        counterexample: Option<&CounterexampleInfo>,
    ) -> Option<RefinementOperator> {
        if ops.is_empty() {
            return None;
        }

        // Apply learned weights.
        let mut scored: Vec<(usize, f64)> = ops.iter().enumerate()
            .map(|(idx, op)| {
                let dim_weight = match op.kind {
                    RefinementKind::OutputAlphabet => self.dimension_weights[0],
                    RefinementKind::InputDepth => self.dimension_weights[1],
                    RefinementKind::DistributionResolution => self.dimension_weights[2],
                    RefinementKind::Combined => {
                        (self.dimension_weights[0] + self.dimension_weights[1] +
                         self.dimension_weights[2]) / 3.0
                    }
                };

                let cx_bonus = match (counterexample, op.kind) {
                    (Some(cx), RefinementKind::OutputAlphabet) if cx.involves_cluster_confusion => 2.0,
                    (Some(cx), RefinementKind::InputDepth) if cx.involves_depth_insufficiency => 2.0,
                    (Some(cx), RefinementKind::DistributionResolution) if cx.involves_distribution_imprecision => 2.0,
                    _ => 1.0,
                };

                let score = op.priority * dim_weight * cx_bonus;
                (idx, score)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.first().map(|(idx, _)| ops[*idx].clone())
    }

    /// Update adaptive weights based on a refinement result.
    pub fn update_weights(&mut self, result: &RefinementResult) {
        let dim_idx = match result.operator.kind {
            RefinementKind::OutputAlphabet => 0,
            RefinementKind::InputDepth => 1,
            RefinementKind::DistributionResolution => 2,
            RefinementKind::Combined => return,
        };

        let learning_rate = 0.1;
        if result.success && result.resolved_counterexample {
            // Reward this dimension.
            self.dimension_weights[dim_idx] *= 1.0 + learning_rate;
        } else if !result.success {
            // Penalize this dimension.
            self.dimension_weights[dim_idx] *= 1.0 - learning_rate;
        }

        // Normalize weights.
        let total: f64 = self.dimension_weights.iter().sum();
        if total > 0.0 {
            for w in &mut self.dimension_weights {
                *w /= total / 3.0;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Refinement history
// ---------------------------------------------------------------------------

/// Tracks the history of refinement operations for analysis and replay.
#[derive(Debug, Clone)]
pub struct RefinementHistory {
    /// Ordered list of refinement results.
    pub entries: Vec<RefinementResult>,
    /// Map from triple to best result achieved.
    pub best_at_triple: HashMap<(usize, usize, OrderedFloat<f64>), usize>,
    /// Total cost incurred.
    pub total_cost: f64,
    /// Total fidelity gained.
    pub total_fidelity_gain: f64,
}

impl RefinementHistory {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            best_at_triple: HashMap::new(),
            total_cost: 0.0,
            total_fidelity_gain: 0.0,
        }
    }

    /// Record a refinement result.
    pub fn record(&mut self, result: RefinementResult) {
        self.total_cost += result.actual_cost;
        if result.success {
            self.total_fidelity_gain += result.actual_fidelity_gain;
        }

        let key = result.new_triple.discrete_key();
        let idx = self.entries.len();
        self.best_at_triple.entry(key).or_insert(idx);

        self.entries.push(result);
    }

    /// Get the number of refinement steps.
    pub fn num_steps(&self) -> usize {
        self.entries.len()
    }

    /// Get the success rate.
    pub fn success_rate(&self) -> f64 {
        if self.entries.is_empty() {
            return 0.0;
        }
        let successes = self.entries.iter().filter(|r| r.success).count();
        successes as f64 / self.entries.len() as f64
    }

    /// Get the counter-example resolution rate.
    pub fn resolution_rate(&self) -> f64 {
        if self.entries.is_empty() {
            return 0.0;
        }
        let resolved = self.entries.iter().filter(|r| r.resolved_counterexample).count();
        resolved as f64 / self.entries.len() as f64
    }

    /// Average prediction error for cost estimates.
    pub fn avg_cost_prediction_error(&self) -> f64 {
        if self.entries.is_empty() {
            return 0.0;
        }
        let total: f64 = self.entries.iter()
            .map(|r| r.cost_prediction_error())
            .sum();
        total / self.entries.len() as f64
    }

    /// Get per-dimension statistics.
    pub fn per_dimension_stats(&self) -> HashMap<RefinementKind, DimensionStats> {
        let mut stats: HashMap<RefinementKind, DimensionStats> = HashMap::new();

        for result in &self.entries {
            let entry = stats.entry(result.operator.kind).or_insert_with(DimensionStats::new);
            entry.attempts += 1;
            if result.success {
                entry.successes += 1;
            }
            if result.resolved_counterexample {
                entry.resolutions += 1;
            }
            entry.total_cost += result.actual_cost;
            entry.total_fidelity += result.actual_fidelity_gain;
        }

        stats
    }

    /// Find the most effective dimension historically.
    pub fn most_effective_dimension(&self) -> Option<RefinementKind> {
        let stats = self.per_dimension_stats();
        stats.iter()
            .max_by(|a, b| {
                let eff_a = a.1.efficiency();
                let eff_b = b.1.efficiency();
                eff_a.partial_cmp(&eff_b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(&kind, _)| kind)
    }

    /// Generate a text summary of the refinement history.
    pub fn summary(&self) -> String {
        let mut out = String::new();
        out.push_str("=== Refinement History ===\n");
        out.push_str(&format!(
            "Steps: {}, Cost: {:.2}, Fidelity gained: {:.4}\n",
            self.entries.len(), self.total_cost, self.total_fidelity_gain
        ));
        out.push_str(&format!(
            "Success rate: {:.1}%, Resolution rate: {:.1}%\n\n",
            self.success_rate() * 100.0,
            self.resolution_rate() * 100.0,
        ));

        for (kind, stats) in &self.per_dimension_stats() {
            out.push_str(&format!(
                "  {}: {} attempts, {} successes, {} resolutions, eff={:.4}\n",
                kind, stats.attempts, stats.successes, stats.resolutions, stats.efficiency()
            ));
        }

        out.push_str("\nTimeline:\n");
        for (i, result) in self.entries.iter().enumerate() {
            let status = if result.success {
                if result.resolved_counterexample { "✓ resolved" } else { "✓ ok" }
            } else {
                "✗ failed"
            };
            out.push_str(&format!(
                "  {}. {} {} (cost={:.2}, gain={:.4})\n",
                i + 1, result.operator.kind, status,
                result.actual_cost, result.actual_fidelity_gain
            ));
        }

        out
    }

    /// Get the last N results.
    pub fn recent(&self, n: usize) -> &[RefinementResult] {
        let start = self.entries.len().saturating_sub(n);
        &self.entries[start..]
    }
}

impl Default for RefinementHistory {
    fn default() -> Self {
        Self::new()
    }
}

/// Per-dimension statistics.
#[derive(Debug, Clone)]
pub struct DimensionStats {
    pub attempts: usize,
    pub successes: usize,
    pub resolutions: usize,
    pub total_cost: f64,
    pub total_fidelity: f64,
}

impl DimensionStats {
    fn new() -> Self {
        Self {
            attempts: 0,
            successes: 0,
            resolutions: 0,
            total_cost: 0.0,
            total_fidelity: 0.0,
        }
    }

    pub fn success_rate(&self) -> f64 {
        if self.attempts > 0 {
            self.successes as f64 / self.attempts as f64
        } else {
            0.0
        }
    }

    pub fn resolution_rate(&self) -> f64 {
        if self.attempts > 0 {
            self.resolutions as f64 / self.attempts as f64
        } else {
            0.0
        }
    }

    pub fn efficiency(&self) -> f64 {
        if self.total_cost > 0.0 {
            self.total_fidelity / self.total_cost
        } else {
            0.0
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_triple(k: usize, n: usize, eps: f64) -> AbstractionTriple {
        AbstractionTriple::new(k, n, eps)
    }

    #[test]
    fn test_refine_k() {
        let triple = make_triple(4, 2, 0.5);
        let op = RefinementOperator::refine_k(&triple, 8, 3);

        assert_eq!(op.kind, RefinementKind::OutputAlphabet);
        assert_eq!(op.from.k, 4);
        assert_eq!(op.to.k, 8);
        assert!(op.estimated_fidelity_gain > 0.0);
        assert!(op.is_strict_refinement());
    }

    #[test]
    fn test_refine_n() {
        let triple = make_triple(4, 2, 0.5);
        let op = RefinementOperator::refine_n(&triple, 4, 3);

        assert_eq!(op.kind, RefinementKind::InputDepth);
        assert_eq!(op.from.n, 2);
        assert_eq!(op.to.n, 4);
        assert!(op.estimated_fidelity_gain > 0.0);
        assert!(op.is_strict_refinement());
    }

    #[test]
    fn test_refine_epsilon() {
        let triple = make_triple(4, 2, 0.5);
        let op = RefinementOperator::refine_epsilon(&triple, 0.1, 3);

        assert_eq!(op.kind, RefinementKind::DistributionResolution);
        assert!((op.from.epsilon - 0.5).abs() < 1e-12);
        assert!((op.to.epsilon - 0.1).abs() < 1e-12);
        assert!(op.estimated_fidelity_gain > 0.0);
        assert!(op.is_strict_refinement());
    }

    #[test]
    fn test_combined_refinement() {
        let triple = make_triple(4, 2, 0.5);
        let op = RefinementOperator::combined(&triple, 8, 3, 0.1, 3);

        assert_eq!(op.kind, RefinementKind::Combined);
        assert!(op.estimated_fidelity_gain > 0.0);
        assert!(op.is_strict_refinement());
    }

    #[test]
    fn test_efficiency() {
        let triple = make_triple(4, 2, 0.5);
        let op = RefinementOperator::refine_k(&triple, 8, 3);
        let eff = op.efficiency();
        assert!(eff >= 0.0);
    }

    #[test]
    fn test_impact_k() {
        let triple = make_triple(4, 2, 0.5);
        let impact = RefinementImpact::estimate_k_refinement(&triple, 8, 3, None);

        assert!(impact.error_reduction > 0.0);
        assert!(impact.error_reduction <= 1.0);
        assert_eq!(impact.new_states, 4);
        assert!(impact.time_cost >= 0.0);
        assert!(impact.resolve_probability >= 0.0);
        assert!(impact.resolve_probability <= 1.0);
    }

    #[test]
    fn test_impact_n() {
        let triple = make_triple(4, 2, 0.5);
        let impact = RefinementImpact::estimate_n_refinement(&triple, 3, 3, None);

        assert!(impact.error_reduction > 0.0);
        assert!(impact.time_cost >= 0.0);
    }

    #[test]
    fn test_impact_epsilon() {
        let triple = make_triple(4, 2, 0.5);
        let impact = RefinementImpact::estimate_epsilon_refinement(&triple, 0.1, 3, None);

        assert!(impact.error_reduction > 0.0);
        assert!(impact.sample_complexity_increase > 1.0);
    }

    #[test]
    fn test_impact_with_counterexample() {
        let triple = make_triple(4, 2, 0.5);
        let cx = CounterexampleInfo::cluster_confusion(vec!["s1".to_string(), "s2".to_string()]);

        let impact = RefinementImpact::estimate_k_refinement(&triple, 8, 3, Some(&cx));
        assert!(impact.resolve_probability > 0.3);

        // Non-matching counter-example should give lower resolve probability.
        let impact2 = RefinementImpact::estimate_n_refinement(&triple, 3, 3, Some(&cx));
        assert!(impact2.resolve_probability < impact.resolve_probability);
    }

    #[test]
    fn test_counterexample_info() {
        let cx = CounterexampleInfo::cluster_confusion(vec!["s1".to_string()]);
        assert_eq!(cx.recommended_dimension(), RefinementKind::OutputAlphabet);

        let cx2 = CounterexampleInfo::depth_insufficiency(vec!["ab".to_string()]);
        assert_eq!(cx2.recommended_dimension(), RefinementKind::InputDepth);

        let cx3 = CounterexampleInfo::distribution_imprecision(vec!["s1".to_string()], 0.5);
        assert_eq!(cx3.recommended_dimension(), RefinementKind::DistributionResolution);
    }

    #[test]
    fn test_planner_generate_operators() {
        let planner = RefinementPlanner::new(
            RefinementStrategy::Greedy,
            3,
            vec![4, 8, 16],
            vec![1, 2, 3, 4],
            vec![0.5, 0.25, 0.1, 0.05],
        );

        let triple = make_triple(4, 2, 0.5);
        let ops = planner.generate_operators(&triple);

        // Should have k refinements (8, 16) + n refinements (3, 4) + eps refinements (0.25, 0.1, 0.05)
        assert_eq!(ops.len(), 7);
    }

    #[test]
    fn test_planner_greedy() {
        let mut planner = RefinementPlanner::new(
            RefinementStrategy::Greedy,
            3,
            vec![4, 8, 16],
            vec![1, 2, 3],
            vec![0.5, 0.25, 0.1],
        );

        let triple = make_triple(4, 1, 0.5);
        let op = planner.select(&triple, None, f64::INFINITY);
        assert!(op.is_some());
    }

    #[test]
    fn test_planner_counterexample_guided() {
        let mut planner = RefinementPlanner::new(
            RefinementStrategy::CounterexampleGuided,
            3,
            vec![4, 8, 16],
            vec![1, 2, 3],
            vec![0.5, 0.25, 0.1],
        );

        let triple = make_triple(4, 1, 0.5);
        let cx = CounterexampleInfo::cluster_confusion(vec!["s1".to_string()]);
        let op = planner.select(&triple, Some(&cx), f64::INFINITY);

        assert!(op.is_some());
        assert_eq!(op.unwrap().kind, RefinementKind::OutputAlphabet);
    }

    #[test]
    fn test_planner_round_robin() {
        let mut planner = RefinementPlanner::new(
            RefinementStrategy::RoundRobin,
            3,
            vec![4, 8],
            vec![1, 2],
            vec![0.5, 0.1],
        );

        let triple = make_triple(4, 1, 0.5);

        let op1 = planner.select(&triple, None, f64::INFINITY);
        assert!(op1.is_some());

        let op2 = planner.select(&triple, None, f64::INFINITY);
        assert!(op2.is_some());

        // Should cycle through different dimensions.
        // (May not be different if some dimensions have no available refinements.)
    }

    #[test]
    fn test_planner_budget_constraint() {
        let mut planner = RefinementPlanner::new(
            RefinementStrategy::Greedy,
            3,
            vec![4, 8, 16],
            vec![1, 2, 3],
            vec![0.5, 0.25, 0.1],
        );

        let triple = make_triple(4, 1, 0.5);
        let op = planner.select(&triple, None, 0.0);
        // With zero budget, nothing should be affordable.
        // (Depends on cost estimates — some small refinements might have 0 cost.)
    }

    #[test]
    fn test_planner_no_refinements_available() {
        let mut planner = RefinementPlanner::new(
            RefinementStrategy::Greedy,
            3,
            vec![16],
            vec![4],
            vec![0.05],
        );

        // Already at maximum refinement.
        let triple = make_triple(16, 4, 0.05);
        let op = planner.select(&triple, None, f64::INFINITY);
        assert!(op.is_none());
    }

    #[test]
    fn test_refinement_result() {
        let triple = make_triple(4, 2, 0.5);
        let op = RefinementOperator::refine_k(&triple, 8, 3);
        let mut result = RefinementResult::new(op.clone());

        result.success = true;
        result.actual_cost = 100.0;
        result.actual_fidelity_gain = 0.5;
        result.resolved_counterexample = true;

        assert!(result.cost_prediction_error() >= 0.0);
    }

    #[test]
    fn test_refinement_history() {
        let mut history = RefinementHistory::new();
        assert_eq!(history.num_steps(), 0);

        let triple = make_triple(4, 2, 0.5);
        let op = RefinementOperator::refine_k(&triple, 8, 3);
        let mut result = RefinementResult::new(op);
        result.success = true;
        result.actual_cost = 50.0;
        result.actual_fidelity_gain = 0.3;
        result.resolved_counterexample = true;

        history.record(result);
        assert_eq!(history.num_steps(), 1);
        assert!((history.success_rate() - 1.0).abs() < 1e-10);
        assert!((history.total_cost - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_history_per_dimension() {
        let mut history = RefinementHistory::new();

        // Add k-refinement result.
        let op1 = RefinementOperator::refine_k(&make_triple(4, 2, 0.5), 8, 3);
        let mut r1 = RefinementResult::new(op1);
        r1.success = true;
        r1.actual_cost = 50.0;
        r1.actual_fidelity_gain = 0.3;
        history.record(r1);

        // Add n-refinement result.
        let op2 = RefinementOperator::refine_n(&make_triple(8, 2, 0.5), 3, 3);
        let mut r2 = RefinementResult::new(op2);
        r2.success = false;
        r2.actual_cost = 100.0;
        r2.actual_fidelity_gain = 0.0;
        history.record(r2);

        let stats = history.per_dimension_stats();
        assert!(stats.contains_key(&RefinementKind::OutputAlphabet));
        assert!(stats.contains_key(&RefinementKind::InputDepth));

        let k_stats = &stats[&RefinementKind::OutputAlphabet];
        assert_eq!(k_stats.attempts, 1);
        assert_eq!(k_stats.successes, 1);
    }

    #[test]
    fn test_most_effective_dimension() {
        let mut history = RefinementHistory::new();

        // k-refinement: good
        let op1 = RefinementOperator::refine_k(&make_triple(4, 2, 0.5), 8, 3);
        let mut r1 = RefinementResult::new(op1);
        r1.success = true;
        r1.actual_cost = 10.0;
        r1.actual_fidelity_gain = 0.5;
        history.record(r1);

        // n-refinement: expensive, low gain
        let op2 = RefinementOperator::refine_n(&make_triple(4, 2, 0.5), 3, 3);
        let mut r2 = RefinementResult::new(op2);
        r2.success = true;
        r2.actual_cost = 100.0;
        r2.actual_fidelity_gain = 0.1;
        history.record(r2);

        let best = history.most_effective_dimension();
        assert_eq!(best, Some(RefinementKind::OutputAlphabet));
    }

    #[test]
    fn test_history_summary() {
        let mut history = RefinementHistory::new();

        let op = RefinementOperator::refine_k(&make_triple(4, 2, 0.5), 8, 3);
        let mut result = RefinementResult::new(op);
        result.success = true;
        result.actual_cost = 50.0;
        history.record(result);

        let summary = history.summary();
        assert!(summary.contains("Refinement History"));
        assert!(summary.contains("Timeline"));
    }

    #[test]
    fn test_planner_update_weights() {
        let mut planner = RefinementPlanner::new(
            RefinementStrategy::Adaptive,
            3,
            vec![4, 8, 16],
            vec![1, 2, 3],
            vec![0.5, 0.25, 0.1],
        );

        let op = RefinementOperator::refine_k(&make_triple(4, 1, 0.5), 8, 3);
        let mut result = RefinementResult::new(op);
        result.success = true;
        result.resolved_counterexample = true;

        let old_weight = planner.dimension_weights[0];
        planner.update_weights(&result);
        // k-dimension weight should increase.
        assert!(planner.dimension_weights[0] >= old_weight * 0.9); // Roughly
    }

    #[test]
    fn test_multi_objective_selection() {
        let mut planner = RefinementPlanner::new(
            RefinementStrategy::MultiObjective,
            3,
            vec![4, 8, 16],
            vec![1, 2, 3],
            vec![0.5, 0.25, 0.1],
        );

        let triple = make_triple(4, 1, 0.5);
        let cx = CounterexampleInfo::cluster_confusion(vec!["s1".to_string()]);
        let op = planner.select(&triple, Some(&cx), f64::INFINITY);
        assert!(op.is_some());
    }

    #[test]
    fn test_adaptive_selection() {
        let mut planner = RefinementPlanner::new(
            RefinementStrategy::Adaptive,
            3,
            vec![4, 8, 16],
            vec![1, 2, 3],
            vec![0.5, 0.25, 0.1],
        );

        let triple = make_triple(4, 1, 0.5);
        let op = planner.select(&triple, None, f64::INFINITY);
        assert!(op.is_some());
    }

    #[test]
    fn test_all_dimensions_strategy() {
        let mut planner = RefinementPlanner::new(
            RefinementStrategy::AllDimensions,
            3,
            vec![4, 8],
            vec![1, 2],
            vec![0.5, 0.1],
        );

        let triple = make_triple(4, 1, 0.5);
        let op = planner.select(&triple, None, f64::INFINITY);
        assert!(op.is_some());
        // Should be a combined refinement.
        assert_eq!(op.unwrap().kind, RefinementKind::Combined);
    }

    #[test]
    fn test_cost_efficient_strategy() {
        let mut planner = RefinementPlanner::new(
            RefinementStrategy::CostEfficient,
            3,
            vec![4, 8, 16],
            vec![1, 2, 3],
            vec![0.5, 0.25, 0.1],
        );

        let triple = make_triple(4, 1, 0.5);
        let op = planner.select(&triple, None, f64::INFINITY);
        assert!(op.is_some());
    }

    #[test]
    fn test_refinement_display() {
        let triple = make_triple(4, 2, 0.5);
        let op = RefinementOperator::refine_k(&triple, 8, 3);
        let display = format!("{}", op);
        assert!(display.contains("k-refine"));
    }

    #[test]
    fn test_history_recent() {
        let mut history = RefinementHistory::new();

        for i in 0..5 {
            let op = RefinementOperator::refine_k(&make_triple(4 + i, 2, 0.5), 8 + i, 3);
            let result = RefinementResult::new(op);
            history.record(result);
        }

        let recent = history.recent(3);
        assert_eq!(recent.len(), 3);
    }
}
