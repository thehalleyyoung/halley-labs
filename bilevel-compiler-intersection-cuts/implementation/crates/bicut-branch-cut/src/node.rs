//! Branch-and-bound node: data, LP solving, warm starting, pruning,
//! integrality checks, bilevel feasibility, reduced-cost fixing, and
//! priority-queue ordering for the BiCut branch-and-cut solver.

use crate::{
    fractionality, BilevelSolution, CompiledBilevelModel, Cut, CutType, LpSolverInterface, NodeId,
    SolutionStatus, BOUND_TOLERANCE, INFINITY_BOUND, INT_TOLERANCE,
};
use bicut_types::*;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::fmt;

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

/// Status of a branch-and-bound node throughout its lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NodeStatus {
    /// Waiting in the queue; LP relaxation has not been solved yet.
    Pending,
    /// LP relaxation solved successfully (status not yet classified).
    Solved,
    /// LP solution satisfies all integrality requirements.
    Integral,
    /// LP solution has at least one fractional integer variable.
    Fractional,
    /// LP relaxation is infeasible at this node.
    Infeasible,
    /// Fathomed: lower bound exceeds incumbent or otherwise dominated.
    Fathomed,
    /// Pruned by a preprocessing rule or domain reduction.
    Pruned,
    /// Children have been created; node is fully processed.
    Branched,
}

impl fmt::Display for NodeStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NodeStatus::Pending => write!(f, "Pending"),
            NodeStatus::Solved => write!(f, "Solved"),
            NodeStatus::Integral => write!(f, "Integral"),
            NodeStatus::Fractional => write!(f, "Fractional"),
            NodeStatus::Infeasible => write!(f, "Infeasible"),
            NodeStatus::Fathomed => write!(f, "Fathomed"),
            NodeStatus::Pruned => write!(f, "Pruned"),
            NodeStatus::Branched => write!(f, "Branched"),
        }
    }
}

impl NodeStatus {
    /// Returns `true` when the node is in a terminal state (no further
    /// processing is required).
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            NodeStatus::Infeasible
                | NodeStatus::Fathomed
                | NodeStatus::Pruned
                | NodeStatus::Branched
                | NodeStatus::Integral
        )
    }

    /// Returns `true` when the node can still be branched on.
    pub fn is_branchable(self) -> bool {
        matches!(self, NodeStatus::Fractional | NodeStatus::Solved)
    }
}

/// Direction of a branching decision on a variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BranchDirection {
    /// x_j <= floor(value)
    Down,
    /// x_j >= ceil(value)
    Up,
}

impl fmt::Display for BranchDirection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BranchDirection::Down => write!(f, "Down"),
            BranchDirection::Up => write!(f, "Up"),
        }
    }
}

// ---------------------------------------------------------------------------
// Supporting structs
// ---------------------------------------------------------------------------

/// A single branching decision recorded in the node's history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchRecord {
    pub variable: VarIndex,
    pub direction: BranchDirection,
    pub bound_value: f64,
    pub parent_lp_obj: f64,
}

impl fmt::Display for BranchRecord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "x{} {} {:.6} (parent obj {:.6})",
            self.variable, self.direction, self.bound_value, self.parent_lp_obj
        )
    }
}

/// Warm-start information saved from a parent node so the LP solver can
/// reuse the basis and start from a nearby point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarmStartInfo {
    pub basis: Vec<BasisStatus>,
    pub primal_solution: Vec<f64>,
    pub dual_solution: Vec<f64>,
}

impl WarmStartInfo {
    /// True when the warm-start data actually contains a usable basis.
    pub fn has_basis(&self) -> bool {
        !self.basis.is_empty()
    }

    /// Number of basic variables in the stored basis.
    pub fn num_basic(&self) -> usize {
        self.basis
            .iter()
            .filter(|&&s| s == BasisStatus::Basic)
            .count()
    }
}

// ---------------------------------------------------------------------------
// BbNode
// ---------------------------------------------------------------------------

/// A single node in the branch-and-bound tree.
///
/// Each node carries its own variable bound tightenings, local cuts, LP
/// solution data, and warm-start information inherited from the parent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BbNode {
    pub id: NodeId,
    pub parent_id: Option<NodeId>,
    pub depth: u32,
    pub status: NodeStatus,
    pub lower_bound: f64,
    pub lp_objective: f64,
    pub lp_solution: Vec<f64>,
    pub lp_dual: Vec<f64>,
    pub lp_basis: Vec<BasisStatus>,
    pub lp_iterations: u64,
    pub var_lower_bounds: Vec<f64>,
    pub var_upper_bounds: Vec<f64>,
    pub branching_history: Vec<BranchRecord>,
    pub local_cuts: Vec<Cut>,
    pub warm_start: Option<WarmStartInfo>,
    pub fractional_vars: Vec<(VarIndex, f64)>,
    pub age: u32,
    pub num_cut_rounds: u32,
    pub is_root: bool,
}

impl BbNode {
    // -------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------

    /// Create the root node from a compiled bilevel model.
    ///
    /// Variable bounds are copied from the LP relaxation; the LP has not been
    /// solved yet so the status is `Pending`.
    pub fn root(model: &CompiledBilevelModel) -> Self {
        let n = model.num_vars;
        let lower: Vec<f64> = (0..n)
            .map(|j| {
                if j < model.lp_relaxation.var_bounds.len() {
                    model.lp_relaxation.var_bounds[j].lower
                } else {
                    0.0
                }
            })
            .collect();
        let upper: Vec<f64> = (0..n)
            .map(|j| {
                if j < model.lp_relaxation.var_bounds.len() {
                    model.lp_relaxation.var_bounds[j].upper
                } else {
                    INFINITY_BOUND
                }
            })
            .collect();
        Self {
            id: 0,
            parent_id: None,
            depth: 0,
            status: NodeStatus::Pending,
            lower_bound: -INFINITY_BOUND,
            lp_objective: INFINITY_BOUND,
            lp_solution: vec![0.0; n],
            lp_dual: Vec::new(),
            lp_basis: Vec::new(),
            lp_iterations: 0,
            var_lower_bounds: lower,
            var_upper_bounds: upper,
            branching_history: Vec::new(),
            local_cuts: Vec::new(),
            warm_start: None,
            fractional_vars: Vec::new(),
            age: 0,
            num_cut_rounds: 0,
            is_root: true,
        }
    }

    /// Create a child node from a branching decision on `variable`.
    ///
    /// The child inherits the parent's local bounds and cuts, tightens the
    /// bound for the branching variable according to `direction`, and stores
    /// warm-start information from the parent.
    pub fn create_child(
        &self,
        child_id: NodeId,
        variable: VarIndex,
        direction: BranchDirection,
        bound_value: f64,
    ) -> Self {
        let n = self.lp_solution.len();
        let mut child = Self {
            id: child_id,
            parent_id: Some(self.id),
            depth: self.depth + 1,
            status: NodeStatus::Pending,
            lower_bound: self.lp_objective,
            lp_objective: INFINITY_BOUND,
            lp_solution: vec![0.0; n],
            lp_dual: Vec::new(),
            lp_basis: Vec::new(),
            lp_iterations: 0,
            var_lower_bounds: self.var_lower_bounds.clone(),
            var_upper_bounds: self.var_upper_bounds.clone(),
            branching_history: self.branching_history.clone(),
            local_cuts: self.local_cuts.clone(),
            warm_start: Some(WarmStartInfo {
                basis: self.lp_basis.clone(),
                primal_solution: self.lp_solution.clone(),
                dual_solution: self.lp_dual.clone(),
            }),
            fractional_vars: Vec::new(),
            age: 0,
            num_cut_rounds: 0,
            is_root: false,
        };

        // Tighten the bound on the branching variable.
        match direction {
            BranchDirection::Down => {
                if variable < child.var_upper_bounds.len() {
                    child.var_upper_bounds[variable] =
                        child.var_upper_bounds[variable].min(bound_value);
                }
            }
            BranchDirection::Up => {
                if variable < child.var_lower_bounds.len() {
                    child.var_lower_bounds[variable] =
                        child.var_lower_bounds[variable].max(bound_value);
                }
            }
        }

        child.branching_history.push(BranchRecord {
            variable,
            direction,
            bound_value,
            parent_lp_obj: self.lp_objective,
        });
        child
    }

    // -------------------------------------------------------------------
    // LP construction & solving
    // -------------------------------------------------------------------

    /// Build the LP relaxation for this node.
    ///
    /// Starts from the model's base LP, overlays the node's tightened variable
    /// bounds, and appends any local cuts as additional constraints.
    pub fn build_node_lp(&self, model: &CompiledBilevelModel) -> LpProblem {
        let base = &model.lp_relaxation;
        let n = base.num_vars;
        let extra_cuts = self.local_cuts.len();
        let m = base.num_constraints + extra_cuts;

        // Tighten variable bounds: take the tighter of base vs. node bounds.
        let mut bounds = base.var_bounds.clone();
        for j in 0..n.min(self.var_lower_bounds.len()) {
            bounds[j].lower = bounds[j].lower.max(self.var_lower_bounds[j]);
        }
        for j in 0..n.min(self.var_upper_bounds.len()) {
            bounds[j].upper = bounds[j].upper.min(self.var_upper_bounds[j]);
        }

        // Build the constraint matrix: base matrix + local cut rows.
        let mut entries = base.a_matrix.entries.clone();
        let mut rhs = base.b_rhs.clone();
        let mut senses = base.senses.clone();

        for (idx, cut) in self.local_cuts.iter().enumerate() {
            let row = base.num_constraints + idx;
            for &(var, coeff) in &cut.coefficients {
                if var < n {
                    entries.push(SparseEntry {
                        row,
                        col: var,
                        value: coeff,
                    });
                }
            }
            rhs.push(cut.rhs);
            senses.push(cut.sense);
        }

        LpProblem {
            direction: base.direction,
            c: base.c.clone(),
            a_matrix: SparseMatrix {
                rows: m,
                cols: n,
                entries,
            },
            b_rhs: rhs,
            senses,
            var_bounds: bounds,
            num_vars: n,
            num_constraints: m,
        }
    }

    /// Solve the LP relaxation at this node.
    ///
    /// Uses warm-start data when available.  Updates the node's LP solution
    /// fields, lower bound, and status.
    pub fn solve_lp(
        &mut self,
        model: &CompiledBilevelModel,
        lp_solver: &dyn LpSolverInterface,
    ) -> LpStatus {
        let node_lp = self.build_node_lp(model);

        let sol = match self.warm_start {
            Some(ref ws) if ws.has_basis() => lp_solver.solve_lp_with_basis(&node_lp, &ws.basis),
            _ => lp_solver.solve_lp(&node_lp),
        };

        self.lp_iterations = sol.iterations;

        match sol.status {
            LpStatus::Optimal => {
                self.lp_objective = sol.objective;
                self.lp_solution = sol.primal;
                self.lp_dual = sol.dual;
                self.lp_basis = sol.basis;
                self.lower_bound = self.lp_objective;
                self.status = NodeStatus::Solved;
            }
            LpStatus::Infeasible => {
                self.status = NodeStatus::Infeasible;
            }
            LpStatus::Unbounded => {
                self.lp_objective = -INFINITY_BOUND;
                self.lower_bound = -INFINITY_BOUND;
                self.status = NodeStatus::Solved;
            }
            LpStatus::IterationLimit | LpStatus::Unknown => {
                self.status = NodeStatus::Fathomed;
            }
        }
        sol.status
    }

    // -------------------------------------------------------------------
    // Integrality & bilevel feasibility
    // -------------------------------------------------------------------

    /// Identify fractional integer variables in the current LP solution.
    ///
    /// Results are stored in `self.fractional_vars` sorted by descending
    /// fractionality so the most fractional variable is first.
    pub fn compute_fractional_vars(&mut self, model: &CompiledBilevelModel, tol: f64) {
        self.fractional_vars.clear();
        for &var in &model.integer_vars {
            if var < self.lp_solution.len() {
                let val = self.lp_solution[var];
                let frac = fractionality(val);
                if frac > tol {
                    self.fractional_vars.push((var, val));
                }
            }
        }
        // Sort by descending fractionality (most fractional first).
        self.fractional_vars.sort_by(|a, b| {
            let fa = fractionality(a.1);
            let fb = fractionality(b.1);
            fb.partial_cmp(&fa).unwrap_or(Ordering::Equal)
        });
    }

    /// Check whether the current LP solution is integer-feasible.
    ///
    /// Internally calls `compute_fractional_vars` and updates `self.status`
    /// to either `Integral` or `Fractional`.
    pub fn check_integrality(&mut self, model: &CompiledBilevelModel, tol: f64) -> bool {
        self.compute_fractional_vars(model, tol);
        if self.fractional_vars.is_empty() {
            self.status = NodeStatus::Integral;
            true
        } else {
            self.status = NodeStatus::Fractional;
            false
        }
    }

    /// Check bilevel feasibility by verifying complementarity conditions.
    ///
    /// For each pair `(s, y)` in the model's complementarity pairs, the
    /// products `s_val * y_val` must be below `tol`.
    pub fn check_bilevel_feasibility(&self, model: &CompiledBilevelModel, tol: f64) -> bool {
        for &(s, y) in &model.complementarity_pairs {
            let sv = if s < self.lp_solution.len() {
                self.lp_solution[s]
            } else {
                0.0
            };
            let yv = if y < self.lp_solution.len() {
                self.lp_solution[y]
            } else {
                0.0
            };
            if sv.abs() > tol && yv.abs() > tol {
                if sv * yv > tol {
                    return false;
                }
            }
        }
        true
    }

    /// Compute the total complementarity violation for diagnostics.
    pub fn complementarity_violation(&self, model: &CompiledBilevelModel) -> f64 {
        let mut total = 0.0_f64;
        for &(s, y) in &model.complementarity_pairs {
            let sv = if s < self.lp_solution.len() {
                self.lp_solution[s].abs()
            } else {
                0.0
            };
            let yv = if y < self.lp_solution.len() {
                self.lp_solution[y].abs()
            } else {
                0.0
            };
            total += sv * yv;
        }
        total
    }

    // -------------------------------------------------------------------
    // Cuts
    // -------------------------------------------------------------------

    /// Append a locally-valid cut to this node.
    pub fn add_local_cut(&mut self, cut: Cut) {
        self.local_cuts.push(cut);
    }

    /// Remove ineffective local cuts whose efficacy at the current LP
    /// solution is below `min_efficacy`.  Returns the number removed.
    pub fn purge_ineffective_cuts(&mut self, min_efficacy: f64) -> usize {
        let sol = &self.lp_solution;
        let before = self.local_cuts.len();
        self.local_cuts.retain(|cut| {
            let lhs: f64 = cut
                .coefficients
                .iter()
                .map(|&(var, coeff)| {
                    if var < sol.len() {
                        coeff * sol[var]
                    } else {
                        0.0
                    }
                })
                .sum();
            let violation = match cut.sense {
                ConstraintSense::Le => lhs - cut.rhs,
                ConstraintSense::Ge => cut.rhs - lhs,
                ConstraintSense::Eq => (lhs - cut.rhs).abs(),
            };
            let norm: f64 = cut
                .coefficients
                .iter()
                .map(|(_, c)| c * c)
                .sum::<f64>()
                .sqrt()
                .max(1e-12);
            let eff = violation / norm;
            eff >= min_efficacy
        });
        before - self.local_cuts.len()
    }

    // -------------------------------------------------------------------
    // Fathoming / pruning
    // -------------------------------------------------------------------

    /// Returns `true` when this node's LP objective is at least as large as
    /// the incumbent, meaning it cannot improve the best known solution.
    pub fn can_fathom(&self, incumbent: f64) -> bool {
        self.lp_objective >= incumbent - BOUND_TOLERANCE
    }

    /// Mark this node as fathomed (dominated by bound).
    pub fn fathom(&mut self) {
        self.status = NodeStatus::Fathomed;
    }

    /// Mark this node as pruned (by preprocessing or domain reduction).
    pub fn prune(&mut self) {
        self.status = NodeStatus::Pruned;
    }

    /// Explicitly set the node status.
    pub fn set_status(&mut self, status: NodeStatus) {
        self.status = status;
    }

    // -------------------------------------------------------------------
    // Solution extraction
    // -------------------------------------------------------------------

    /// Extract a feasible bilevel solution from this node, returning `None`
    /// unless the node is marked `Integral`.
    pub fn extract_solution(&self, model: &CompiledBilevelModel) -> Option<BilevelSolution> {
        if self.status != NodeStatus::Integral {
            return None;
        }
        let bilevel_feas = self.check_bilevel_feasibility(model, INT_TOLERANCE);
        Some(BilevelSolution {
            values: self.lp_solution.clone(),
            objective: self.lp_objective,
            status: if bilevel_feas {
                SolutionStatus::Feasible
            } else {
                SolutionStatus::Feasible
            },
            is_bilevel_feasible: bilevel_feas,
            gap: 0.0,
            node_count: 0,
            time_secs: 0.0,
        })
    }

    /// Extract a rounded solution: round all integer variables to their
    /// nearest integer values and check feasibility.  Returns `None` if
    /// any rounded value falls outside its bounds.
    pub fn extract_rounded_solution(
        &self,
        model: &CompiledBilevelModel,
    ) -> Option<BilevelSolution> {
        if self.lp_solution.is_empty() {
            return None;
        }
        let n = self.lp_solution.len();
        let mut rounded = self.lp_solution.clone();

        for &var in &model.integer_vars {
            if var < n {
                rounded[var] = rounded[var].round();
                // Clamp to bounds.
                let lo = if var < self.var_lower_bounds.len() {
                    self.var_lower_bounds[var]
                } else {
                    0.0
                };
                let hi = if var < self.var_upper_bounds.len() {
                    self.var_upper_bounds[var]
                } else {
                    INFINITY_BOUND
                };
                if rounded[var] < lo - BOUND_TOLERANCE || rounded[var] > hi + BOUND_TOLERANCE {
                    return None;
                }
                rounded[var] = rounded[var].max(lo).min(hi);
            }
        }

        // Compute the objective for the rounded solution.
        let obj: f64 = model
            .lp_relaxation
            .c
            .iter()
            .zip(rounded.iter())
            .map(|(c, x)| c * x)
            .sum();

        let bilevel_feas = {
            let mut ok = true;
            for &(s, y) in &model.complementarity_pairs {
                let sv = if s < rounded.len() { rounded[s] } else { 0.0 };
                let yv = if y < rounded.len() { rounded[y] } else { 0.0 };
                if sv.abs() > INT_TOLERANCE && yv.abs() > INT_TOLERANCE {
                    if sv * yv > INT_TOLERANCE {
                        ok = false;
                        break;
                    }
                }
            }
            ok
        };

        Some(BilevelSolution {
            values: rounded,
            objective: obj,
            status: SolutionStatus::Feasible,
            is_bilevel_feasible: bilevel_feas,
            gap: 0.0,
            node_count: 0,
            time_secs: 0.0,
        })
    }

    // -------------------------------------------------------------------
    // Warm start
    // -------------------------------------------------------------------

    /// Snapshot the current LP solution into a `WarmStartInfo`.
    pub fn get_warm_start(&self) -> WarmStartInfo {
        WarmStartInfo {
            basis: self.lp_basis.clone(),
            primal_solution: self.lp_solution.clone(),
            dual_solution: self.lp_dual.clone(),
        }
    }

    /// Replace the warm-start data for this node.
    pub fn set_warm_start(&mut self, ws: WarmStartInfo) {
        self.warm_start = Some(ws);
    }

    /// Discard warm-start data (e.g. after cuts invalidate the basis).
    pub fn clear_warm_start(&mut self) {
        self.warm_start = None;
    }

    // -------------------------------------------------------------------
    // Bound queries
    // -------------------------------------------------------------------

    /// Returns `true` when every variable's lower bound ≤ upper bound.
    pub fn bounds_consistent(&self) -> bool {
        let n = self.var_lower_bounds.len().min(self.var_upper_bounds.len());
        (0..n).all(|j| self.var_lower_bounds[j] <= self.var_upper_bounds[j] + BOUND_TOLERANCE)
    }

    /// Simple priority score: lower bound (ascending is better).
    pub fn priority_score(&self) -> f64 {
        self.lower_bound
    }

    /// Hybrid priority that blends lower bound with a depth bonus.
    ///
    /// Lower scores are preferred.  `depth_weight > 0` encourages
    /// depth-first exploration by subtracting `depth_weight * depth`.
    pub fn hybrid_score(&self, depth_weight: f64) -> f64 {
        self.lower_bound - depth_weight * self.depth as f64
    }

    /// Count variables whose domain has been fixed to a single value.
    pub fn num_fixed_vars(&self) -> usize {
        let n = self.var_lower_bounds.len().min(self.var_upper_bounds.len());
        (0..n)
            .filter(|&j| {
                (self.var_upper_bounds[j] - self.var_lower_bounds[j]).abs() < BOUND_TOLERANCE
            })
            .count()
    }

    /// Width of the domain for variable `var`.
    pub fn var_domain_width(&self, var: VarIndex) -> f64 {
        if var < self.var_lower_bounds.len() && var < self.var_upper_bounds.len() {
            (self.var_upper_bounds[var] - self.var_lower_bounds[var]).max(0.0)
        } else {
            INFINITY_BOUND
        }
    }

    /// Total domain volume (product of widths, but capped to avoid overflow).
    /// Uses a log-space sum instead for numerical stability.
    pub fn log_domain_volume(&self) -> f64 {
        let n = self.var_lower_bounds.len().min(self.var_upper_bounds.len());
        let mut log_vol = 0.0_f64;
        for j in 0..n {
            let width = (self.var_upper_bounds[j] - self.var_lower_bounds[j]).max(1e-20);
            if width < INFINITY_BOUND {
                log_vol += width.ln();
            }
        }
        log_vol
    }

    /// Increment the age counter (used for aging-based node selection).
    pub fn increment_age(&mut self) {
        self.age += 1;
    }

    // -------------------------------------------------------------------
    // Reduced-cost fixing
    // -------------------------------------------------------------------

    /// Apply reduced-cost fixing to tighten variable bounds.
    ///
    /// For each variable at its lower (upper) bound, if the reduced cost
    /// exceeds the gap to the incumbent, the opposite bound can be fixed.
    /// Returns the number of variables fixed.
    pub fn apply_reduced_cost_fixing(
        &mut self,
        reduced_costs: &[f64],
        incumbent: f64,
        tol: f64,
    ) -> usize {
        let gap = incumbent - self.lp_objective;
        if gap <= tol {
            return 0;
        }
        let n = self
            .var_lower_bounds
            .len()
            .min(self.var_upper_bounds.len())
            .min(reduced_costs.len());
        let mut fixed = 0usize;
        for j in 0..n {
            let rc = reduced_costs[j];
            let val = self.lp_solution.get(j).copied().unwrap_or(0.0);

            // Variable at lower bound with large positive reduced cost.
            if (val - self.var_lower_bounds[j]).abs() < tol && rc > gap + tol {
                self.var_upper_bounds[j] = self.var_lower_bounds[j];
                fixed += 1;
                continue;
            }
            // Variable at upper bound with large negative reduced cost.
            if (val - self.var_upper_bounds[j]).abs() < tol && (-rc) > gap + tol {
                self.var_lower_bounds[j] = self.var_upper_bounds[j];
                fixed += 1;
            }
        }
        fixed
    }

    /// Tighten bounds on a single variable.  Returns `true` if the domain
    /// became empty (infeasible).
    pub fn tighten_var_bounds(&mut self, var: VarIndex, new_lower: f64, new_upper: f64) -> bool {
        if var < self.var_lower_bounds.len() {
            self.var_lower_bounds[var] = self.var_lower_bounds[var].max(new_lower);
        }
        if var < self.var_upper_bounds.len() {
            self.var_upper_bounds[var] = self.var_upper_bounds[var].min(new_upper);
        }
        if var < self.var_lower_bounds.len() && var < self.var_upper_bounds.len() {
            self.var_lower_bounds[var] > self.var_upper_bounds[var] + BOUND_TOLERANCE
        } else {
            false
        }
    }

    /// Apply simple bound propagation based on the model's constraint matrix.
    ///
    /// Iterates through each constraint and tightens singleton-variable
    /// bounds where possible.  Returns the number of bounds tightened.
    pub fn propagate_bounds(&mut self, model: &CompiledBilevelModel) -> usize {
        let n = model.num_vars;
        let base = &model.lp_relaxation;
        let m = base.num_constraints;
        let mut tightened = 0usize;

        // Group entries by row for efficient processing.
        let mut row_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); m];
        for entry in &base.a_matrix.entries {
            if entry.row < m && entry.col < n {
                row_entries[entry.row].push((entry.col, entry.value));
            }
        }

        for i in 0..m {
            if row_entries[i].len() != 1 {
                continue;
            }
            let (col, coeff) = row_entries[i][0];
            if coeff.abs() < 1e-12 {
                continue;
            }
            let rhs = base.b_rhs[i];

            // Single-variable constraint: coeff * x_col sense rhs
            let implied = rhs / coeff;
            match base.senses[i] {
                ConstraintSense::Le => {
                    if coeff > 0.0 {
                        // x_col <= implied
                        if col < self.var_upper_bounds.len()
                            && implied < self.var_upper_bounds[col] - BOUND_TOLERANCE
                        {
                            self.var_upper_bounds[col] = implied;
                            tightened += 1;
                        }
                    } else {
                        // x_col >= implied (coeff < 0 flips)
                        if col < self.var_lower_bounds.len()
                            && implied > self.var_lower_bounds[col] + BOUND_TOLERANCE
                        {
                            self.var_lower_bounds[col] = implied;
                            tightened += 1;
                        }
                    }
                }
                ConstraintSense::Ge => {
                    if coeff > 0.0 {
                        if col < self.var_lower_bounds.len()
                            && implied > self.var_lower_bounds[col] + BOUND_TOLERANCE
                        {
                            self.var_lower_bounds[col] = implied;
                            tightened += 1;
                        }
                    } else {
                        if col < self.var_upper_bounds.len()
                            && implied < self.var_upper_bounds[col] - BOUND_TOLERANCE
                        {
                            self.var_upper_bounds[col] = implied;
                            tightened += 1;
                        }
                    }
                }
                ConstraintSense::Eq => {
                    if col < self.var_lower_bounds.len() {
                        self.var_lower_bounds[col] = self.var_lower_bounds[col].max(implied);
                    }
                    if col < self.var_upper_bounds.len() {
                        self.var_upper_bounds[col] = self.var_upper_bounds[col].min(implied);
                    }
                    tightened += 1;
                }
            }
        }
        tightened
    }
}

// ---------------------------------------------------------------------------
// Display
// ---------------------------------------------------------------------------

impl fmt::Display for BbNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Node {} [d={}, {}, lb={:.6}, obj={:.6}, cuts={}, frac={}]",
            self.id,
            self.depth,
            self.status,
            self.lower_bound,
            self.lp_objective,
            self.local_cuts.len(),
            self.fractional_vars.len(),
        )
    }
}

// ---------------------------------------------------------------------------
// Ordering for priority queue (min-heap: lower bound ascending)
// ---------------------------------------------------------------------------

impl PartialEq for BbNode {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for BbNode {}

impl PartialOrd for BbNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Best-first ordering: nodes with *smaller* lower bound come first.
/// Ties are broken by preferring *deeper* nodes (larger depth) to encourage
/// finding integer solutions sooner.
impl Ord for BbNode {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse because BinaryHeap is a max-heap; we want smallest LB first.
        other
            .lower_bound
            .partial_cmp(&self.lower_bound)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.depth.cmp(&self.depth))
    }
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Compute reduced costs: `rc_j = c_j - Σ_i dual_i * A_{ij}`.
///
/// Reduced costs indicate how much the objective would change per unit
/// increase of variable `j` from its bound.  Used for reduced-cost fixing.
pub fn compute_reduced_costs(model: &CompiledBilevelModel, dual: &[f64]) -> Vec<f64> {
    let n = model.num_vars;
    let m = model.lp_relaxation.num_constraints;

    // Start with the objective coefficients.
    let mut rc = model.lp_relaxation.c.clone();
    if rc.len() < n {
        rc.resize(n, 0.0);
    }

    // Subtract dual^T * A column by column.
    for entry in &model.lp_relaxation.a_matrix.entries {
        if entry.row < m && entry.col < n && entry.row < dual.len() {
            rc[entry.col] -= dual[entry.row] * entry.value;
        }
    }
    rc
}

/// Compute reduced costs from a custom constraint matrix and dual vector.
pub fn compute_reduced_costs_custom(
    obj: &[f64],
    matrix: &SparseMatrix,
    dual: &[f64],
    num_vars: usize,
) -> Vec<f64> {
    let mut rc: Vec<f64> = if obj.len() >= num_vars {
        obj[..num_vars].to_vec()
    } else {
        let mut v = obj.to_vec();
        v.resize(num_vars, 0.0);
        v
    };

    for entry in &matrix.entries {
        if entry.col < num_vars && entry.row < dual.len() {
            rc[entry.col] -= dual[entry.row] * entry.value;
        }
    }
    rc
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_model() -> CompiledBilevelModel {
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
        let mut model = CompiledBilevelModel::new(bilevel);
        model.integer_vars = vec![0];
        model.complementarity_pairs = vec![(0, 1)];
        model
    }

    #[test]
    fn test_root_node_creation() {
        let m = make_test_model();
        let r = BbNode::root(&m);
        assert_eq!(r.id, 0);
        assert!(r.is_root);
        assert_eq!(r.depth, 0);
        assert!(r.parent_id.is_none());
        assert_eq!(r.status, NodeStatus::Pending);
        assert_eq!(r.lp_solution.len(), m.num_vars);
        assert!(r.branching_history.is_empty());
        assert!(r.bounds_consistent());
    }

    #[test]
    fn test_child_down_branch() {
        let m = make_test_model();
        let root = BbNode::root(&m);
        let child = root.create_child(1, 0, BranchDirection::Down, 3.0);
        assert_eq!(child.depth, 1);
        assert_eq!(child.parent_id, Some(0));
        assert!(!child.is_root);
        assert!(child.var_upper_bounds[0] <= 3.0 + BOUND_TOLERANCE);
        assert_eq!(child.branching_history.len(), 1);
        assert_eq!(child.branching_history[0].variable, 0);
        assert_eq!(child.branching_history[0].direction, BranchDirection::Down);
        assert!(child.warm_start.is_some());
    }

    #[test]
    fn test_child_up_branch() {
        let m = make_test_model();
        let root = BbNode::root(&m);
        let child = root.create_child(2, 0, BranchDirection::Up, 4.0);
        assert!(child.var_lower_bounds[0] >= 4.0 - BOUND_TOLERANCE);
        assert_eq!(
            child.branching_history.last().unwrap().direction,
            BranchDirection::Up
        );
    }

    #[test]
    fn test_bounds_consistency_check() {
        let m = make_test_model();
        let mut node = BbNode::root(&m);
        assert!(node.bounds_consistent());

        // Make bounds inconsistent.
        if !node.var_lower_bounds.is_empty() && !node.var_upper_bounds.is_empty() {
            node.var_lower_bounds[0] = 100.0;
            node.var_upper_bounds[0] = 1.0;
            assert!(!node.bounds_consistent());
        }
    }

    #[test]
    fn test_fractional_variable_detection() {
        let m = make_test_model();
        let mut node = BbNode::root(&m);
        node.lp_solution = vec![2.5, 1.0];
        node.compute_fractional_vars(&m, 1e-6);
        assert_eq!(node.fractional_vars.len(), 1);
        assert_eq!(node.fractional_vars[0].0, 0);

        // Now with an integer value.
        node.lp_solution = vec![3.0, 1.0];
        node.compute_fractional_vars(&m, 1e-6);
        assert!(node.fractional_vars.is_empty());
    }

    #[test]
    fn test_integrality_check() {
        let m = make_test_model();
        let mut node = BbNode::root(&m);

        // Integral case.
        node.lp_solution = vec![3.0, 1.5];
        assert!(node.check_integrality(&m, 1e-6));
        assert_eq!(node.status, NodeStatus::Integral);

        // Fractional case.
        node.lp_solution = vec![2.7, 1.0];
        assert!(!node.check_integrality(&m, 1e-6));
        assert_eq!(node.status, NodeStatus::Fractional);
    }

    #[test]
    fn test_fathom_by_bound() {
        let m = make_test_model();
        let mut node = BbNode::root(&m);
        node.lp_objective = 10.0;
        assert!(node.can_fathom(9.0));
        assert!(node.can_fathom(10.0));
        assert!(!node.can_fathom(11.0));

        node.fathom();
        assert_eq!(node.status, NodeStatus::Fathomed);
    }

    #[test]
    fn test_node_ordering_for_priority_queue() {
        use std::collections::BinaryHeap;

        let m = make_test_model();
        let mut n1 = BbNode::root(&m);
        n1.lower_bound = 5.0;
        n1.id = 1;
        let mut n2 = BbNode::root(&m);
        n2.lower_bound = 3.0;
        n2.id = 2;
        let mut n3 = BbNode::root(&m);
        n3.lower_bound = 7.0;
        n3.id = 3;

        let mut heap = BinaryHeap::new();
        heap.push(n1);
        heap.push(n2);
        heap.push(n3);

        // Best-first: smallest lower bound is popped first from the max-heap.
        let first = heap.pop().unwrap();
        assert_eq!(first.id, 2); // lb = 3.0
        let second = heap.pop().unwrap();
        assert_eq!(second.id, 1); // lb = 5.0
        let third = heap.pop().unwrap();
        assert_eq!(third.id, 3); // lb = 7.0
    }

    #[test]
    fn test_local_cut_management() {
        let m = make_test_model();
        let mut node = BbNode::root(&m);
        assert!(node.local_cuts.is_empty());

        node.add_local_cut(Cut::new(
            vec![(0, 1.0)],
            5.0,
            ConstraintSense::Le,
            CutType::Gomory,
            false,
        ));
        assert_eq!(node.local_cuts.len(), 1);

        node.add_local_cut(Cut::new(
            vec![(0, 2.0), (1, -1.0)],
            3.0,
            ConstraintSense::Le,
            CutType::BilevelIntersection,
            true,
        ));
        assert_eq!(node.local_cuts.len(), 2);
    }

    #[test]
    fn test_reduced_cost_computation() {
        let m = make_test_model();
        let dual = vec![0.0; m.lp_relaxation.num_constraints];
        let rc = compute_reduced_costs(&m, &dual);
        // With zero duals, reduced costs equal objective coefficients.
        assert_eq!(rc.len(), m.num_vars);
        for (j, &c) in m.lp_relaxation.c.iter().enumerate() {
            assert!((rc[j] - c).abs() < 1e-10);
        }
    }
}
