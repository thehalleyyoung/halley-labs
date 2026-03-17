//! Bounding operations for the bilevel branch-and-cut solver.
//!
//! This module manages global primal and dual bounds, tracks the incumbent
//! solution, and implements several bound-tightening techniques:
//!
//! * **Reduced-cost fixing** – exploits LP dual information to fix or tighten
//!   variable bounds in a subtree.
//! * **Branching propagation** – enforces the bound changes implied by the
//!   branching decision that created a node.
//! * **Feasibility-based bound tightening (FBBT)** – iterates over constraints
//!   to derive implied bounds on individual variables.
//! * **Complementarity propagation** – if one variable in a complementarity
//!   pair has a positive lower bound the other must be zero.

use crate::node::{BbNode, BranchDirection, NodeStatus};
use crate::{compute_gap, CompiledBilevelModel, BOUND_TOLERANCE, INFINITY_BOUND};
use bicut_types::{ConstraintSense, VarBound, VarIndex};

// ---------------------------------------------------------------------------
// BoundManager
// ---------------------------------------------------------------------------

/// Manages global primal (upper) and dual (lower) bounds during branch-and-cut.
///
/// The primal bound equals the objective of the best known feasible solution
/// (the *incumbent*). The dual bound is derived from the LP relaxation values
/// of all open nodes in the tree.
#[derive(Debug, Clone)]
pub struct BoundManager {
    /// Global lower bound (dual bound) — minimum LP bound across open nodes.
    pub global_lower_bound: f64,
    /// Global upper bound (primal bound) — incumbent objective value.
    pub global_upper_bound: f64,
    /// Variable values of the current best feasible solution, if any.
    pub incumbent_solution: Option<Vec<f64>>,
    /// Objective value of the current incumbent.
    pub incumbent_objective: f64,
    /// Cumulative count of variables fixed by reduced-cost arguments.
    pub reduced_cost_fixings: u64,
    /// Cumulative count of individual bound tightenings through propagation.
    pub bound_propagations: u64,
}

impl BoundManager {
    /// Create a new `BoundManager` with trivially loose initial bounds.
    pub fn new() -> Self {
        Self {
            global_lower_bound: -INFINITY_BOUND,
            global_upper_bound: INFINITY_BOUND,
            incumbent_solution: None,
            incumbent_objective: INFINITY_BOUND,
            reduced_cost_fixings: 0,
            bound_propagations: 0,
        }
    }

    /// Update the global lower (dual) bound if `bound` strictly improves it.
    ///
    /// Returns `true` when the bound was updated.
    pub fn update_lower_bound(&mut self, bound: f64) -> bool {
        if bound > self.global_lower_bound + BOUND_TOLERANCE {
            self.global_lower_bound = bound;
            true
        } else {
            false
        }
    }

    /// Update the incumbent (upper / primal bound) if `obj` is better than the
    /// current best.
    ///
    /// Returns `true` when a new incumbent was accepted.
    pub fn update_upper_bound(&mut self, obj: f64, solution: Vec<f64>) -> bool {
        if obj < self.incumbent_objective - BOUND_TOLERANCE {
            self.incumbent_objective = obj;
            self.global_upper_bound = obj;
            self.incumbent_solution = Some(solution);
            true
        } else {
            false
        }
    }

    /// Compute the current relative optimality gap.
    pub fn get_gap(&self) -> f64 {
        compute_gap(self.global_upper_bound, self.global_lower_bound)
    }

    /// Check whether the gap is within the specified tolerance, meaning the
    /// solver can declare optimality.
    pub fn is_optimal(&self, tol: f64) -> bool {
        self.get_gap() <= tol
    }

    /// Determine whether `node` can be pruned by bound.
    ///
    /// A node is prunable when its lower bound meets or exceeds the incumbent
    /// objective (within tolerance), so no descendant can improve the solution.
    pub fn can_prune_node(&self, node: &BbNode) -> bool {
        node.lower_bound >= self.incumbent_objective - BOUND_TOLERANCE
    }

    // -----------------------------------------------------------------------
    // Reduced-cost fixing
    // -----------------------------------------------------------------------

    /// Apply reduced-cost fixing at a node.
    ///
    /// For a minimisation LP, if variable *j* sits at its lower bound in the
    /// LP solution and its reduced cost `rc_j` satisfies
    /// `rc_j > UB − node_LB`, then the variable can be fixed at its lower
    /// bound in the subtree.  Analogously for variables at their upper bound
    /// with `rc_j < −(UB − node_LB)`.  A partial tightening is also applied
    /// when the reduced cost is large but does not dominate the gap entirely.
    ///
    /// Returns the number of variables whose bounds were tightened.
    pub fn apply_reduced_cost_fixing(&mut self, node: &mut BbNode, reduced_costs: &[f64]) -> usize {
        let gap = self.incumbent_objective - node.lower_bound;
        if gap <= BOUND_TOLERANCE {
            return 0;
        }

        let num_rc = reduced_costs.len();
        let mut fixings = 0usize;

        for j in 0..num_rc {
            let rc = reduced_costs[j];
            let (cur_lb, cur_ub) = effective_var_bounds(node, j);

            // Already fixed — nothing to do.
            if (cur_ub - cur_lb).abs() < BOUND_TOLERANCE {
                continue;
            }

            if rc > gap + BOUND_TOLERANCE {
                // Variable at lower bound; rc exceeds gap → fix at lower bound.
                if cur_ub > cur_lb + BOUND_TOLERANCE {
                    set_node_bound(
                        node,
                        j,
                        VarBound {
                            lower: cur_lb,
                            upper: cur_lb,
                        },
                    );
                    fixings += 1;
                }
            } else if rc < -(gap + BOUND_TOLERANCE) {
                // Variable at upper bound; |rc| exceeds gap → fix at upper bound.
                if cur_lb < cur_ub - BOUND_TOLERANCE {
                    set_node_bound(
                        node,
                        j,
                        VarBound {
                            lower: cur_ub,
                            upper: cur_ub,
                        },
                    );
                    fixings += 1;
                }
            } else if rc > BOUND_TOLERANCE && gap > BOUND_TOLERANCE {
                // Partial tightening of upper bound.
                let new_ub = cur_lb + gap / rc;
                if new_ub < cur_ub - BOUND_TOLERANCE {
                    set_node_bound(
                        node,
                        j,
                        VarBound {
                            lower: cur_lb,
                            upper: new_ub,
                        },
                    );
                    fixings += 1;
                }
            } else if rc < -BOUND_TOLERANCE && gap > BOUND_TOLERANCE {
                // Partial tightening of lower bound.
                let new_lb = cur_ub + gap / rc; // rc < 0 ⇒ gap/rc < 0
                if new_lb > cur_lb + BOUND_TOLERANCE {
                    set_node_bound(
                        node,
                        j,
                        VarBound {
                            lower: new_lb,
                            upper: cur_ub,
                        },
                    );
                    fixings += 1;
                }
            }
        }

        self.reduced_cost_fixings += fixings as u64;
        fixings
    }

    // -----------------------------------------------------------------------
    // Branching propagation
    // -----------------------------------------------------------------------

    /// Propagate variable-bound changes implied by the branching decision that
    /// created `node`.
    ///
    /// * *Down* branch on x_j with bound b → x_j ≤ b.
    /// * *Up* branch on x_j with bound b   → x_j ≥ b.
    ///
    /// Returns `true` if at least one bound was tightened.
    pub fn propagate_bounds_from_branching(
        &mut self,
        node: &mut BbNode,
        _model: &CompiledBilevelModel,
    ) -> bool {
        let branch = match node.branching_history.last() {
            Some(b) => b.clone(),
            None => return false,
        };

        let var = branch.variable;
        let (cur_lb, cur_ub) = effective_var_bounds(node, var);
        let mut changed = false;

        match branch.direction {
            BranchDirection::Down => {
                let new_ub = branch.bound_value;
                if new_ub < cur_ub - BOUND_TOLERANCE {
                    set_node_bound(
                        node,
                        var,
                        VarBound {
                            lower: cur_lb,
                            upper: new_ub,
                        },
                    );
                    self.bound_propagations += 1;
                    changed = true;
                }
            }
            BranchDirection::Up => {
                let new_lb = branch.bound_value;
                if new_lb > cur_lb + BOUND_TOLERANCE {
                    set_node_bound(
                        node,
                        var,
                        VarBound {
                            lower: new_lb,
                            upper: cur_ub,
                        },
                    );
                    self.bound_propagations += 1;
                    changed = true;
                }
            }
        }

        changed
    }

    // -----------------------------------------------------------------------
    // Constraint-based bound tightening (FBBT)
    // -----------------------------------------------------------------------

    /// Tighten variable bounds using constraint information (FBBT).
    ///
    /// Runs two forward passes over all constraints in the LP relaxation to
    /// derive implied bounds on individual variables. Bounds that are tighter
    /// than the current node bounds are applied.
    ///
    /// Returns the total number of individual bound tightenings.
    pub fn propagate_constraint_bounds(
        &mut self,
        node: &mut BbNode,
        model: &CompiledBilevelModel,
    ) -> usize {
        let lp = &model.lp_relaxation;
        let n = lp.num_vars;
        let m = lp.num_constraints;

        // Snapshot current variable bounds (model defaults + node overrides).
        let mut lb = vec![-INFINITY_BOUND; n];
        let mut ub = vec![INFINITY_BOUND; n];
        for j in 0..n.min(lp.var_bounds.len()) {
            lb[j] = lp.var_bounds[j].lower;
            ub[j] = lp.var_bounds[j].upper;
        }
        for j in 0..n.min(node.var_lower_bounds.len()) {
            lb[j] = lb[j].max(node.var_lower_bounds[j]);
        }
        for j in 0..n.min(node.var_upper_bounds.len()) {
            ub[j] = ub[j].min(node.var_upper_bounds[j]);
        }

        // Build per-row sparse representation.
        let mut rows: Vec<Vec<(VarIndex, f64)>> = vec![Vec::new(); m];
        for entry in &lp.a_matrix.entries {
            if entry.row < m && entry.col < n && entry.value.abs() > BOUND_TOLERANCE {
                rows[entry.row].push((entry.col, entry.value));
            }
        }

        let mut total_tightenings = 0usize;

        // Two FBBT passes for better propagation.
        for _pass in 0..2 {
            let mut pass_tightenings = 0usize;

            for i in 0..m {
                let sense = if i < lp.senses.len() {
                    lp.senses[i]
                } else {
                    ConstraintSense::Le
                };
                let rhs = if i < lp.b_rhs.len() { lp.b_rhs[i] } else { 0.0 };

                let implied = compute_implied_bounds(&rows[i], rhs, sense, &lb, &ub);

                for (var, new_lb, new_ub) in implied {
                    if new_lb > lb[var] + BOUND_TOLERANCE {
                        lb[var] = new_lb;
                        pass_tightenings += 1;
                    }
                    if new_ub < ub[var] - BOUND_TOLERANCE {
                        ub[var] = new_ub;
                        pass_tightenings += 1;
                    }
                }
            }

            total_tightenings += pass_tightenings;
            if pass_tightenings == 0 {
                break; // Fixed-point reached.
            }
        }

        // Write tightened bounds back to the node.
        for j in 0..n {
            let orig_lb = if j < lp.var_bounds.len() {
                lp.var_bounds[j].lower
            } else {
                -INFINITY_BOUND
            };
            let orig_ub = if j < lp.var_bounds.len() {
                lp.var_bounds[j].upper
            } else {
                INFINITY_BOUND
            };
            if lb[j] > orig_lb + BOUND_TOLERANCE || ub[j] < orig_ub - BOUND_TOLERANCE {
                set_node_bound(
                    node,
                    j,
                    VarBound {
                        lower: lb[j],
                        upper: ub[j],
                    },
                );
            }
        }

        self.bound_propagations += total_tightenings as u64;
        total_tightenings
    }

    // -----------------------------------------------------------------------
    // Global dual bound from the tree
    // -----------------------------------------------------------------------

    /// Compute the global lower (dual) bound from the set of open nodes.
    ///
    /// The global LB equals the minimum LP bound across all nodes that are
    /// still [`NodeStatus::Active`].  If no active node exists the current
    /// global lower bound is returned unchanged.
    pub fn dual_bound_improvement(&self, nodes: &[BbNode]) -> f64 {
        let mut best = INFINITY_BOUND;
        for node in nodes {
            if !node.status.is_terminal() && node.lower_bound < best {
                best = node.lower_bound;
            }
        }
        if best >= INFINITY_BOUND {
            self.global_lower_bound
        } else {
            best
        }
    }

    /// Compute the minimum over a slice of node-bound values.
    ///
    /// This lightweight helper is status-agnostic — it simply returns the
    /// smallest value in the slice, or [`INFINITY_BOUND`] if the slice is
    /// empty.
    pub fn compute_lower_bound_from_tree(node_bounds: &[f64]) -> f64 {
        node_bounds.iter().copied().fold(INFINITY_BOUND, f64::min)
    }

    // -----------------------------------------------------------------------
    // Combined tightening
    // -----------------------------------------------------------------------

    /// Run **all** available bound-tightening techniques on `node` and return
    /// the total number of individual tightenings.
    ///
    /// The order is: branching propagation → constraint FBBT →
    /// complementarity propagation.
    pub fn try_bound_tightening(
        &mut self,
        node: &mut BbNode,
        model: &CompiledBilevelModel,
    ) -> usize {
        let mut total = 0usize;

        // 1. Branching propagation.
        if self.propagate_bounds_from_branching(node, model) {
            total += 1;
        }

        // 2. Constraint-based FBBT.
        total += self.propagate_constraint_bounds(node, model);

        // 3. Complementarity propagation.
        total += propagate_complementarity_bounds(node, model);

        total
    }
}

impl Default for BoundManager {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Retrieve the effective bounds for variable `var` at the given node by
/// scanning local bound-change overrides (latest entry wins).  Falls back to
/// `[0, INFINITY_BOUND]` when no override exists.
fn effective_var_bounds(node: &BbNode, var: VarIndex) -> (f64, f64) {
    let lb = if var < node.var_lower_bounds.len() {
        node.var_lower_bounds[var]
    } else {
        0.0
    };
    let ub = if var < node.var_upper_bounds.len() {
        node.var_upper_bounds[var]
    } else {
        INFINITY_BOUND
    };
    (lb, ub)
}

/// Insert or tighten a bound-change entry for `var` at `node`.
///
/// If the variable already has an entry the new bounds are intersected with
/// the existing ones (lower raised, upper lowered).
fn set_node_bound(node: &mut BbNode, var: VarIndex, new_bound: VarBound) {
    if var < node.var_lower_bounds.len() {
        node.var_lower_bounds[var] = node.var_lower_bounds[var].max(new_bound.lower);
    }
    if var < node.var_upper_bounds.len() {
        node.var_upper_bounds[var] = node.var_upper_bounds[var].min(new_bound.upper);
    }
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Compute implied variable bounds from a single linear constraint.
///
/// Given a sparse constraint row `a_row` (variable-index / coefficient pairs),
/// a right-hand side `rhs`, a constraint sense, and current lower/upper
/// bounds on every variable, this function performs one round of
/// feasibility-based bound tightening (FBBT).
///
/// # Returns
///
/// A vector of `(variable_index, implied_lb, implied_ub)` for every variable
/// whose implied bound is strictly tighter than the current bound.
pub fn compute_implied_bounds(
    a_row: &[(VarIndex, f64)],
    rhs: f64,
    sense: ConstraintSense,
    current_lb: &[f64],
    current_ub: &[f64],
) -> Vec<(VarIndex, f64, f64)> {
    if a_row.is_empty() {
        return Vec::new();
    }

    // ------------------------------------------------------------------
    // Step 1: aggregate row-wide min / max activity and track unbounded
    //         contributions.
    // ------------------------------------------------------------------
    let mut min_activity = 0.0_f64;
    let mut max_activity = 0.0_f64;
    let mut min_unbounded_count: u32 = 0;
    let mut max_unbounded_count: u32 = 0;
    let mut min_unbounded_var: Option<VarIndex> = None;
    let mut max_unbounded_var: Option<VarIndex> = None;

    for &(var, coeff) in a_row {
        if coeff.abs() < BOUND_TOLERANCE {
            continue;
        }
        let lo = safe_lb(current_lb, var);
        let hi = safe_ub(current_ub, var);

        let (c_min, c_max) = contrib_range(coeff, lo, hi);

        if c_min <= -INFINITY_BOUND + 1.0 {
            min_unbounded_count += 1;
            min_unbounded_var = Some(var);
        } else {
            min_activity += c_min;
        }

        if c_max >= INFINITY_BOUND - 1.0 {
            max_unbounded_count += 1;
            max_unbounded_var = Some(var);
        } else {
            max_activity += c_max;
        }
    }

    // ------------------------------------------------------------------
    // Step 2: for each variable derive implied bounds.
    // ------------------------------------------------------------------
    let mut results = Vec::new();

    for &(target, coeff) in a_row {
        if coeff.abs() < BOUND_TOLERANCE {
            continue;
        }
        let lo = safe_lb(current_lb, target);
        let hi = safe_ub(current_ub, target);
        let (t_min, t_max) = contrib_range(coeff, lo, hi);

        let target_min_is_inf = t_min <= -INFINITY_BOUND + 1.0;
        let target_max_is_inf = t_max >= INFINITY_BOUND - 1.0;

        // Rest-min: total min minus target's min contribution.
        let (rest_min, rest_min_ok) = rest_act(
            min_activity,
            t_min,
            min_unbounded_count,
            min_unbounded_var,
            target,
            target_min_is_inf,
        );
        // Rest-max: total max minus target's max contribution.
        let (rest_max, rest_max_ok) = rest_act(
            max_activity,
            t_max,
            max_unbounded_count,
            max_unbounded_var,
            target,
            target_max_is_inf,
        );

        let mut new_lb = -INFINITY_BOUND;
        let mut new_ub = INFINITY_BOUND;

        // Le (or Eq-Le part): a·x ≤ rhs  ⟹  a_j·x_j ≤ rhs − min_rest
        if matches!(sense, ConstraintSense::Le | ConstraintSense::Eq) && rest_min_ok {
            let residual = rhs - rest_min;
            if coeff > BOUND_TOLERANCE {
                let candidate = residual / coeff;
                if candidate < new_ub {
                    new_ub = candidate;
                }
            } else if coeff < -BOUND_TOLERANCE {
                let candidate = residual / coeff;
                if candidate > new_lb {
                    new_lb = candidate;
                }
            }
        }

        // Ge (or Eq-Ge part): a·x ≥ rhs  ⟹  a_j·x_j ≥ rhs − max_rest
        if matches!(sense, ConstraintSense::Ge | ConstraintSense::Eq) && rest_max_ok {
            let residual = rhs - rest_max;
            if coeff > BOUND_TOLERANCE {
                let candidate = residual / coeff;
                if candidate > new_lb {
                    new_lb = candidate;
                }
            } else if coeff < -BOUND_TOLERANCE {
                let candidate = residual / coeff;
                if candidate < new_ub {
                    new_ub = candidate;
                }
            }
        }

        let improved_lb = new_lb > lo + BOUND_TOLERANCE;
        let improved_ub = new_ub < hi - BOUND_TOLERANCE;
        if improved_lb || improved_ub {
            results.push((
                target,
                if improved_lb { new_lb } else { lo },
                if improved_ub { new_ub } else { hi },
            ));
        }
    }

    results
}

/// Propagate bounds using complementarity conditions.
///
/// For each complementarity pair `(u, v)` in the model the relation
/// `u · v = 0` must hold at every feasible point.  Therefore, if the lower
/// bound on `u` is strictly positive, `v` must be fixed to zero — and vice
/// versa.
///
/// Returns the number of individual bound tightenings applied.
pub fn propagate_complementarity_bounds(node: &mut BbNode, model: &CompiledBilevelModel) -> usize {
    let mut tightenings = 0usize;

    for &(u, v) in &model.complementarity_pairs {
        // u > 0  ⟹  v = 0
        let (u_lb, _) = effective_var_bounds(node, u);
        if u_lb > BOUND_TOLERANCE {
            let (v_lb, v_ub) = effective_var_bounds(node, v);
            if v_ub > BOUND_TOLERANCE || v_lb < -BOUND_TOLERANCE {
                set_node_bound(
                    node,
                    v,
                    VarBound {
                        lower: 0.0,
                        upper: 0.0,
                    },
                );
                tightenings += 1;
            }
        }

        // v > 0  ⟹  u = 0  (re-read v bounds after possible change above)
        let (v_lb_now, _) = effective_var_bounds(node, v);
        if v_lb_now > BOUND_TOLERANCE {
            let (u_lb_now, u_ub_now) = effective_var_bounds(node, u);
            if u_ub_now > BOUND_TOLERANCE || u_lb_now < -BOUND_TOLERANCE {
                set_node_bound(
                    node,
                    u,
                    VarBound {
                        lower: 0.0,
                        upper: 0.0,
                    },
                );
                tightenings += 1;
            }
        }
    }

    tightenings
}

// ---------------------------------------------------------------------------
// Internal arithmetic helpers
// ---------------------------------------------------------------------------

/// Safe lower-bound lookup; returns `-INFINITY_BOUND` for out-of-range vars.
#[inline]
fn safe_lb(current_lb: &[f64], var: VarIndex) -> f64 {
    if var < current_lb.len() {
        current_lb[var]
    } else {
        -INFINITY_BOUND
    }
}

/// Safe upper-bound lookup; returns `INFINITY_BOUND` for out-of-range vars.
#[inline]
fn safe_ub(current_ub: &[f64], var: VarIndex) -> f64 {
    if var < current_ub.len() {
        current_ub[var]
    } else {
        INFINITY_BOUND
    }
}

/// `(min, max)` of `coeff * x` for `x ∈ [lo, hi]`.
#[inline]
fn contrib_range(coeff: f64, lo: f64, hi: f64) -> (f64, f64) {
    if coeff > 0.0 {
        (coeff * lo, coeff * hi)
    } else {
        (coeff * hi, coeff * lo)
    }
}

/// Compute rest-activity (min or max) **excluding** one target variable.
///
/// * `total` – aggregated finite activity over all variables.
/// * `target_contrib` – the target variable's contribution to `total`.
/// * `unbounded_count` – how many variables had ±∞ contributions.
/// * `unbounded_var` – the single variable responsible when exactly one is ∞.
/// * `target` – the variable we are excluding.
/// * `target_is_inf` – whether the target itself was an unbounded contributor.
///
/// Returns `(rest_activity, is_finite)`.
fn rest_act(
    total: f64,
    target_contrib: f64,
    unbounded_count: u32,
    unbounded_var: Option<VarIndex>,
    target: VarIndex,
    target_is_inf: bool,
) -> (f64, bool) {
    if target_is_inf {
        let remaining = unbounded_count - 1;
        if remaining == 0 {
            (total, true)
        } else {
            (0.0, false)
        }
    } else if unbounded_count == 0 {
        (total - target_contrib, true)
    } else if unbounded_count == 1 && unbounded_var == Some(target) {
        // The only unbounded variable is the target itself (shouldn't reach
        // here since target_is_inf would be true, but guard anyway).
        (total - target_contrib, true)
    } else {
        (0.0, false)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::{BbNode, BranchDirection, BranchRecord, NodeStatus};
    use crate::CompiledBilevelModel;
    use bicut_types::*;

    // -- Test helpers -------------------------------------------------------

    /// Build a trivial model with `n` variables, no constraints.
    fn make_simple_model(n: usize) -> CompiledBilevelModel {
        let bilevel = BilevelProblem {
            upper_obj_c_x: vec![1.0; n],
            upper_obj_c_y: vec![],
            lower_obj_c: vec![],
            lower_a: SparseMatrix::new(0, 0),
            lower_b: vec![],
            lower_linking_b: SparseMatrix::new(0, n),
            upper_constraints_a: SparseMatrix::new(0, n),
            upper_constraints_b: vec![],
            num_upper_vars: n,
            num_lower_vars: 0,
            num_lower_constraints: 0,
            num_upper_constraints: 0,
        };
        let mut model = CompiledBilevelModel::new(bilevel);
        for j in 0..n {
            model.lp_relaxation.var_bounds[j] = VarBound {
                lower: 0.0,
                upper: 10.0,
            };
        }
        model
    }

    /// Build a model with one constraint: x0 + x1 ≤ 5, x0,x1 ∈ [0,10].
    fn make_constrained_model() -> CompiledBilevelModel {
        let bilevel = BilevelProblem {
            upper_obj_c_x: vec![1.0, 1.0],
            upper_obj_c_y: vec![],
            lower_obj_c: vec![],
            lower_a: SparseMatrix::new(0, 0),
            lower_b: vec![],
            lower_linking_b: SparseMatrix::new(0, 2),
            upper_constraints_a: SparseMatrix::new(1, 2),
            upper_constraints_b: vec![5.0],
            num_upper_vars: 2,
            num_lower_vars: 0,
            num_lower_constraints: 0,
            num_upper_constraints: 1,
        };
        let mut model = CompiledBilevelModel::new(bilevel);
        model.lp_relaxation.a_matrix = SparseMatrix::new(1, 2);
        model.lp_relaxation.a_matrix.add_entry(0, 0, 1.0);
        model.lp_relaxation.a_matrix.add_entry(0, 1, 1.0);
        model.lp_relaxation.b_rhs = vec![5.0];
        model.lp_relaxation.senses = vec![ConstraintSense::Le];
        model.lp_relaxation.var_bounds = vec![
            VarBound {
                lower: 0.0,
                upper: 10.0,
            },
            VarBound {
                lower: 0.0,
                upper: 10.0,
            },
        ];
        model.lp_relaxation.num_vars = 2;
        model.lp_relaxation.num_constraints = 1;
        model
    }

    /// Convenience: create a root node with explicit variable bounds.
    fn root_with_bounds(bounds: &[(VarIndex, f64, f64)]) -> BbNode {
        let n = bounds.iter().map(|&(v, _, _)| v + 1).max().unwrap_or(0);
        let model = make_simple_model(n);
        let mut node = BbNode::root(&model);
        for &(var, lo, hi) in bounds {
            if var < node.var_lower_bounds.len() {
                node.var_lower_bounds[var] = lo;
            }
            if var < node.var_upper_bounds.len() {
                node.var_upper_bounds[var] = hi;
            }
        }
        node
    }

    // -- Tests --------------------------------------------------------------

    #[test]
    fn test_bound_manager_initial_state() {
        let bm = BoundManager::new();
        assert!(bm.global_lower_bound < -1e18);
        assert!(bm.global_upper_bound > 1e18);
        assert!(bm.incumbent_solution.is_none());
        assert_eq!(bm.reduced_cost_fixings, 0);
        assert_eq!(bm.bound_propagations, 0);
        assert!(bm.get_gap() > 1.0);
    }

    #[test]
    fn test_update_bounds_and_gap() {
        let mut bm = BoundManager::new();

        // Lower bound updates.
        assert!(bm.update_lower_bound(5.0));
        assert!((bm.global_lower_bound - 5.0).abs() < 1e-10);
        assert!(!bm.update_lower_bound(5.0), "same value → no update");
        assert!(!bm.update_lower_bound(3.0), "worse value → no update");
        assert!(bm.update_lower_bound(7.0), "better value → update");

        // Upper bound (incumbent) updates.
        assert!(bm.update_upper_bound(100.0, vec![1.0, 2.0]));
        assert!((bm.incumbent_objective - 100.0).abs() < 1e-10);
        assert!(bm.incumbent_solution.is_some());
        assert!(bm.update_upper_bound(50.0, vec![3.0]));
        assert!((bm.incumbent_objective - 50.0).abs() < 1e-10);
        assert!(!bm.update_upper_bound(80.0, vec![9.0]), "worse → no update");

        // Gap: (50 − 7) / |50| = 0.86
        let gap = bm.get_gap();
        assert!((gap - 0.86).abs() < 1e-8);
        assert!(!bm.is_optimal(0.01));
        assert!(bm.is_optimal(0.9));
    }

    #[test]
    fn test_can_prune_node() {
        let mut bm = BoundManager::new();
        bm.update_upper_bound(10.0, vec![1.0]);

        let model = make_simple_model(1);
        let mut node = BbNode::root(&model);
        node.lower_bound = 5.0;
        assert!(!bm.can_prune_node(&node));

        node.lower_bound = 10.0;
        assert!(bm.can_prune_node(&node));

        node.lower_bound = 15.0;
        assert!(bm.can_prune_node(&node));
    }

    #[test]
    fn test_reduced_cost_fixing() {
        let mut bm = BoundManager::new();
        bm.update_upper_bound(10.0, vec![0.0, 0.0, 0.0]);

        let mut node = root_with_bounds(&[(0, 0.0, 5.0), (1, 0.0, 5.0), (2, 0.0, 5.0)]);
        node.lower_bound = 8.0;
        // gap = 10 − 8 = 2
        // rc[0] = 3.0 > gap=2 → fix x0 at lower bound (0)
        // rc[1] = 1.0 < gap   → partial: new_ub = 0 + 2/1 = 2
        // rc[2] = 0.0         → no change
        let rc = vec![3.0, 1.0, 0.0];
        let count = bm.apply_reduced_cost_fixing(&mut node, &rc);
        assert!(count >= 2, "expected ≥ 2 fixings, got {}", count);

        let (lb0, ub0) = effective_var_bounds(&node, 0);
        assert!((ub0 - lb0).abs() < 1e-8, "x0 should be fixed at 0");

        let (_, ub1) = effective_var_bounds(&node, 1);
        assert!(ub1 <= 2.0 + 1e-8, "x1 ub should be ≤ 2, got {}", ub1);
    }

    #[test]
    fn test_propagate_bounds_from_branching() {
        let model = make_simple_model(3);
        let mut bm = BoundManager::new();

        let root = root_with_bounds(&[(0, 0.0, 10.0), (1, 0.0, 10.0)]);
        // Down branch on x0 at bound 3.0 → ub should end up ≤ 3.0
        let mut child = root.create_child(1, 0, BranchDirection::Down, 3.0);
        // create_child already tightens the bound; propagation should confirm.
        bm.propagate_bounds_from_branching(&mut child, &model);
        let (_, ub) = effective_var_bounds(&child, 0);
        assert!(ub <= 3.0 + 1e-8, "Down branch: ub ≤ 3, got {}", ub);

        // Up branch on x1 at bound 4.0 → lb should end up ≥ 4.0
        let mut child_up = root.create_child(2, 1, BranchDirection::Up, 4.0);
        bm.propagate_bounds_from_branching(&mut child_up, &model);
        let (lb1, _) = effective_var_bounds(&child_up, 1);
        assert!(lb1 >= 4.0 - 1e-8, "Up branch: lb ≥ 4, got {}", lb1);
    }

    #[test]
    fn test_propagate_constraint_bounds() {
        let model = make_constrained_model();
        let mut bm = BoundManager::new();
        let mut node = root_with_bounds(&[(0, 0.0, 10.0), (1, 0.0, 10.0)]);

        let count = bm.propagate_constraint_bounds(&mut node, &model);
        assert!(count > 0, "should tighten at least one bound");

        let (_, ub0) = effective_var_bounds(&node, 0);
        let (_, ub1) = effective_var_bounds(&node, 1);
        assert!(ub0 <= 5.0 + 1e-8, "x0 ub ≤ 5, got {}", ub0);
        assert!(ub1 <= 5.0 + 1e-8, "x1 ub ≤ 5, got {}", ub1);
    }

    #[test]
    fn test_compute_implied_bounds_le() {
        // 2x0 + 3x1 ≤ 12, x0,x1 ∈ [0,10]
        let row = vec![(0, 2.0), (1, 3.0)];
        let lb = vec![0.0, 0.0];
        let ub = vec![10.0, 10.0];

        let implied = compute_implied_bounds(&row, 12.0, ConstraintSense::Le, &lb, &ub);
        assert!(!implied.is_empty());
        for &(var, _nlb, nub) in &implied {
            match var {
                0 => assert!(nub <= 6.0 + 1e-8, "x0 ub ≤ 6, got {}", nub),
                1 => assert!(nub <= 4.0 + 1e-8, "x1 ub ≤ 4, got {}", nub),
                _ => panic!("unexpected variable"),
            }
        }

        // Equality: x0 + x1 = 3, both in [0,5] → ub tightened to 3
        let row_eq = vec![(0, 1.0), (1, 1.0)];
        let lb_eq = vec![0.0, 0.0];
        let ub_eq = vec![5.0, 5.0];
        let implied_eq = compute_implied_bounds(&row_eq, 3.0, ConstraintSense::Eq, &lb_eq, &ub_eq);
        for &(_, _nlb, nub) in &implied_eq {
            assert!(nub <= 3.0 + 1e-8);
        }

        // Empty row → empty result.
        let empty: Vec<(VarIndex, f64)> = vec![];
        assert!(compute_implied_bounds(&empty, 0.0, ConstraintSense::Le, &[], &[]).is_empty());
    }

    #[test]
    fn test_dual_bound_improvement_and_tree_lb() {
        let bm = BoundManager::new();
        let model = make_simple_model(1);
        let mut n1 = BbNode::root(&model);
        n1.lower_bound = 5.0;
        let mut n2 = BbNode::root(&model);
        n2.id = 1;
        n2.lower_bound = 3.0;
        let mut n3 = BbNode::root(&model);
        n3.id = 2;
        n3.lower_bound = 7.0;
        n3.status = NodeStatus::Fathomed;

        let dual = bm.dual_bound_improvement(&[n1, n2, n3]);
        assert!(
            (dual - 3.0).abs() < 1e-10,
            "min of active = 3, got {}",
            dual
        );

        // Static helper is status-agnostic.
        let tree_lb = BoundManager::compute_lower_bound_from_tree(&[5.0, 3.0, 7.0]);
        assert!((tree_lb - 3.0).abs() < 1e-10);

        // Empty slice → INFINITY_BOUND.
        assert!(BoundManager::compute_lower_bound_from_tree(&[]) >= INFINITY_BOUND - 1.0);
    }
}
