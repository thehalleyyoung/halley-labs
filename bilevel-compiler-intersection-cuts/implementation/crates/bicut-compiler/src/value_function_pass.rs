//! Value function reformulation pass.
//!
//! Replaces the optimality condition  y ∈ argmin { c^T y : Ay ≤ b + Bx, y ≥ 0 }
//! with the equivalent system:
//!
//! 1. **Primal feasibility**:  Ay ≤ b + Bx,  y ≥ 0
//! 2. **Value function cut**:  c^T y ≤ φ(x)
//!
//! where φ(x) = min { c^T y : Ay ≤ b + Bx, y ≥ 0 } is represented as the
//! pointwise maximum of affine pieces: φ(x) = max_k { α_k^T x + β_k }.
//!
//! An epigraph variable `t` satisfying `t ≥ α_k^T x + β_k` for each piece k
//! replaces φ(x) in the linear formulation, yielding `c^T y ≤ t`.
//!
//! Optional McCormick envelopes linearise any bilinear terms that arise
//! when upper-level constraints couple x and y.

use crate::pipeline::{
    IndicatorConstraint, MilpConstraint, MilpProblem, MilpVariable, Sos1Set, VarType,
};
use crate::*;

use bicut_lp::{solve_lp, SimplexSolver};
use bicut_types::{
    AffineFunction, BilevelProblem, ConstraintSense, LpProblem, LpStatus, OptDirection,
    SparseEntry, SparseMatrix, VarBound, DEFAULT_TOLERANCE,
};
use bicut_value_function::{AffinePiece, ExactLpOracle, PiecewiseLinearVF, ValueFunctionOracle};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the value-function reformulation pass.
#[derive(Debug, Clone)]
pub struct ValueFunctionConfig {
    /// Numerical tolerance for zero comparisons.
    pub tolerance: f64,
    /// Maximum number of affine pieces kept in the PWL approximation.
    pub max_pieces: usize,
    /// Number of sample points used to build the initial PWL approximation.
    pub sampling_points: usize,
    /// Whether to add McCormick relaxations for bilinear x·y terms.
    pub add_mccormick: bool,
    /// Number of partitions per McCormick variable.
    pub mccormick_partitions: usize,
    /// Bounds on each upper-level variable x_i: (lower, upper).
    pub x_bounds: Vec<(f64, f64)>,
}

impl Default for ValueFunctionConfig {
    fn default() -> Self {
        Self {
            tolerance: DEFAULT_TOLERANCE,
            max_pieces: 200,
            sampling_points: 50,
            add_mccormick: false,
            mccormick_partitions: 4,
            x_bounds: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------

/// Output produced by the value-function reformulation pass.
#[derive(Debug, Clone)]
pub struct ValueFunctionResult {
    /// The single-level MILP encoding.
    pub milp: MilpProblem,
    /// How many PWL pieces were used.
    pub num_pieces: usize,
    /// The piecewise-linear value function.
    pub value_function: PiecewiseLinearVF,
    /// How many McCormick auxiliary variables were introduced.
    pub num_mccormick_vars: usize,
    /// Non-fatal warnings gathered during compilation.
    pub warnings: Vec<String>,
}

// ---------------------------------------------------------------------------
// Variable layout
// ---------------------------------------------------------------------------

/// Maps variable groups to consecutive index ranges in the MILP.
#[derive(Debug, Clone)]
pub struct VFVarLayout {
    pub x_start: usize,
    pub x_count: usize,
    pub y_start: usize,
    pub y_count: usize,
    /// Epigraph variable index for φ(x).
    pub phi_var: usize,
    pub mccormick_start: usize,
    pub mccormick_count: usize,
    pub total_vars: usize,
}

impl VFVarLayout {
    /// Build a layout for a given bilevel problem and number of McCormick
    /// auxiliary variables.
    pub fn new(problem: &BilevelProblem, num_mccormick: usize) -> Self {
        let x_start = 0;
        let x_count = problem.num_upper_vars;
        let y_start = x_count;
        let y_count = problem.num_lower_vars;
        let phi_var = y_start + y_count;
        let mccormick_start = phi_var + 1;
        let mccormick_count = num_mccormick;
        let total_vars = mccormick_start + mccormick_count;
        Self {
            x_start,
            x_count,
            y_start,
            y_count,
            phi_var,
            mccormick_start,
            mccormick_count,
            total_vars,
        }
    }

    /// Index of the i-th upper-level (x) variable in the MILP.
    pub fn x_idx(&self, i: usize) -> usize {
        self.x_start + i
    }

    /// Index of the i-th lower-level (y) variable in the MILP.
    pub fn y_idx(&self, i: usize) -> usize {
        self.y_start + i
    }
}

// ---------------------------------------------------------------------------
// McCormick relaxation
// ---------------------------------------------------------------------------

/// Linearisation of the bilinear product  w = x · y  over a box.
///
/// The four McCormick envelope inequalities are:
/// ```text
///   w ≥ x_lo · y + x · y_lo − x_lo · y_lo
///   w ≥ x_hi · y + x · y_hi − x_hi · y_hi
///   w ≤ x_hi · y + x · y_lo − x_hi · y_lo
///   w ≤ x_lo · y + x · y_hi − x_lo · y_hi
/// ```
#[derive(Debug, Clone)]
pub struct McCormickRelaxation {
    pub x_var: usize,
    pub y_var: usize,
    pub aux_var: usize,
    pub x_lo: f64,
    pub x_hi: f64,
    pub y_lo: f64,
    pub y_hi: f64,
}

impl McCormickRelaxation {
    /// Generate the four McCormick envelope constraints as
    /// `(coeffs, sense, rhs)` triples.
    ///
    /// Each `coeffs` vec contains `(var_idx, coeff)` pairs.
    pub fn constraints(&self) -> Vec<(Vec<(usize, f64)>, ConstraintSense, f64)> {
        let (x, y, w) = (self.x_var, self.y_var, self.aux_var);
        let (xl, xh, yl, yh) = (self.x_lo, self.x_hi, self.y_lo, self.y_hi);

        vec![
            // w ≥ xl·y + x·yl − xl·yl  ⟹  w − xl·y − yl·x ≥ −xl·yl
            (
                vec![(w, 1.0), (y, -xl), (x, -yl)],
                ConstraintSense::Ge,
                -xl * yl,
            ),
            // w ≥ xh·y + x·yh − xh·yh  ⟹  w − xh·y − yh·x ≥ −xh·yh
            (
                vec![(w, 1.0), (y, -xh), (x, -yh)],
                ConstraintSense::Ge,
                -xh * yh,
            ),
            // w ≤ xh·y + x·yl − xh·yl  ⟹  w − xh·y − yl·x ≤ −xh·yl
            (
                vec![(w, 1.0), (y, -xh), (x, -yl)],
                ConstraintSense::Le,
                -xh * yl,
            ),
            // w ≤ xl·y + x·yh − xl·yh  ⟹  w − xl·y − yh·x ≤ −xl·yh
            (
                vec![(w, 1.0), (y, -xl), (x, -yh)],
                ConstraintSense::Le,
                -xl * yh,
            ),
        ]
    }
}

// ---------------------------------------------------------------------------
// Value function pass
// ---------------------------------------------------------------------------

/// The value-function reformulation pass.
pub struct ValueFunctionPass {
    config: ValueFunctionConfig,
}

impl ValueFunctionPass {
    pub fn new(config: ValueFunctionConfig) -> Self {
        Self { config }
    }

    // ── public entry point ─────────────────────────────────────────

    /// Apply the value-function reformulation to produce a single-level MILP.
    pub fn apply(&self, problem: &BilevelProblem) -> Result<ValueFunctionResult, CompilerError> {
        if problem.num_lower_vars == 0 {
            return Err(CompilerError::InvalidProblem(
                "lower level has zero variables".into(),
            ));
        }

        let mut warnings: Vec<String> = Vec::new();

        // 1. Compute piecewise-linear value function
        let vf = self.compute_value_function(problem)?;
        let num_pieces = vf.num_pieces();
        if num_pieces == 0 {
            return Err(CompilerError::Numerical(
                "value function has zero pieces".into(),
            ));
        }

        // 2. Detect bilinear terms and count McCormick auxiliaries
        let bilinear_terms = self.detect_bilinear_terms(problem);
        let num_mccormick = if self.config.add_mccormick {
            bilinear_terms.len()
        } else {
            0
        };

        // 3. Build variable layout
        let layout = VFVarLayout::new(problem, num_mccormick);

        // 4. Build MILP
        let mut milp = MilpProblem::new("value_function_reformulation");
        milp.sense = OptDirection::Minimize;

        // Add all variables
        self.add_upper_variables(&mut milp, problem, &layout);
        self.add_lower_variables(&mut milp, problem, &layout);
        self.add_epigraph_variable(&mut milp, problem, &vf, &layout);
        let mccormick_relaxations = if self.config.add_mccormick {
            self.add_mccormick_for_bilinear(&mut milp, &bilinear_terms, &layout)
        } else {
            Vec::new()
        };

        // 5. Build objective
        self.build_upper_objective(&mut milp, problem, &layout);

        // 6. Add constraints
        self.add_primal_feasibility(&mut milp, problem, &layout);
        self.add_value_function_constraint(&mut milp, problem, &vf, &layout);
        self.add_upper_constraints(&mut milp, problem, &layout);

        // 7. McCormick envelope constraints
        for mc in &mccormick_relaxations {
            for (coeffs, sense, rhs) in mc.constraints() {
                let mut con = MilpConstraint::new("mccormick", sense, rhs);
                for (idx, coeff) in &coeffs {
                    con.add_term(*idx, *coeff);
                }
                milp.add_constraint(con);
            }
        }

        if num_pieces > self.config.max_pieces {
            warnings.push(format!(
                "PWL approximation has {} pieces, exceeding max_pieces={}",
                num_pieces, self.config.max_pieces,
            ));
        }

        Ok(ValueFunctionResult {
            milp,
            num_pieces,
            value_function: vf,
            num_mccormick_vars: num_mccormick,
            warnings,
        })
    }

    // ── value function computation ─────────────────────────────────

    /// Compute a piecewise-linear approximation of φ(x) by sampling x-space,
    /// solving the lower-level LP at each sample, and constructing affine
    /// under-estimators from the dual solutions.
    pub fn compute_value_function(
        &self,
        problem: &BilevelProblem,
    ) -> Result<PiecewiseLinearVF, CompilerError> {
        let dim = problem.num_upper_vars;
        let mut pwl = PiecewiseLinearVF::new(dim);
        let samples = self.sample_x_points(problem);

        let mut feasible_count = 0usize;
        for x in &samples {
            if let Some((obj, dual)) = self.solve_lower_level_at(problem, x) {
                let piece =
                    Self::construct_affine_piece(&dual, &problem.lower_b, &problem.lower_linking_b);
                pwl.add_piece(piece);
                feasible_count += 1;
            }
        }

        if feasible_count == 0 {
            return Err(CompilerError::Numerical(
                "all sample points yielded infeasible lower-level problems".into(),
            ));
        }

        // Trim to max_pieces by keeping the ones with the most diverse gradients
        if pwl.num_pieces() > self.config.max_pieces {
            self.trim_pieces(&mut pwl);
        }

        Ok(pwl)
    }

    // ── primal feasibility ─────────────────────────────────────────

    /// Add primal feasibility constraints: Ay ≤ b + Bx, y ≥ 0.
    ///
    /// The y ≥ 0 bounds are already encoded via variable lower bounds.
    pub fn add_primal_feasibility(
        &self,
        milp: &mut MilpProblem,
        problem: &BilevelProblem,
        layout: &VFVarLayout,
    ) {
        let m = problem.num_lower_constraints;
        let n_y = problem.num_lower_vars;
        let n_x = problem.num_upper_vars;

        // Build dense row representation for A and B to emit one constraint
        // per lower-level row.
        for row in 0..m {
            let mut con = MilpConstraint::new(
                &format!("primal_feas_{}", row),
                ConstraintSense::Le,
                problem.lower_b[row],
            );

            // A_row · y terms
            for entry in &problem.lower_a.entries {
                if entry.row == row {
                    con.add_term(layout.y_idx(entry.col), entry.value);
                }
            }

            // -B_row · x  (move Bx to the left: Ay - Bx ≤ b)
            for entry in &problem.lower_linking_b.entries {
                if entry.row == row {
                    con.add_term(layout.x_idx(entry.col), -entry.value);
                }
            }

            milp.add_constraint(con);
        }
    }

    // ── value function / epigraph constraints ──────────────────────

    /// Add the value-function constraint block:
    ///
    /// *  c^T y ≤ t              (optimality cut)
    /// *  t ≥ α_k^T x + β_k     for each piece k  (epigraph)
    pub fn add_value_function_constraint(
        &self,
        milp: &mut MilpProblem,
        problem: &BilevelProblem,
        vf: &PiecewiseLinearVF,
        layout: &VFVarLayout,
    ) {
        // c^T y ≤ t
        let mut opt_cut = MilpConstraint::new("vf_optimality", ConstraintSense::Le, 0.0);
        for (j, &cj) in problem.lower_obj_c.iter().enumerate() {
            opt_cut.add_term(layout.y_idx(j), cj);
        }
        opt_cut.add_term(layout.phi_var, -1.0);
        milp.add_constraint(opt_cut);

        // t ≥ α_k^T x + β_k  for each piece k
        for (k, piece) in vf.pieces.iter().enumerate() {
            let mut con = MilpConstraint::new(
                &format!("vf_piece_{}", k),
                ConstraintSense::Ge,
                piece.constant,
            );
            con.add_term(layout.phi_var, 1.0);
            for (i, &ai) in piece.coefficients.iter().enumerate() {
                // t ≥ α_k^T x + β_k  ⟹  t − α_k^T x ≥ β_k
                con.add_term(layout.x_idx(i), -ai);
            }
            milp.add_constraint(con);
        }
    }

    // ── epigraph variable ──────────────────────────────────────────

    /// Add the epigraph variable `t` (≈ φ(x)) with appropriate bounds.
    pub fn add_epigraph_variable(
        &self,
        milp: &mut MilpProblem,
        problem: &BilevelProblem,
        vf: &PiecewiseLinearVF,
        layout: &VFVarLayout,
    ) {
        // Compute a finite lower bound from the minimum piece constant
        let lb = vf
            .pieces
            .iter()
            .map(|p| {
                let min_contrib: f64 = p
                    .coefficients
                    .iter()
                    .enumerate()
                    .map(|(i, &a)| {
                        let (lo, hi) = self.x_bound(i, problem);
                        if a >= 0.0 {
                            a * lo
                        } else {
                            a * hi
                        }
                    })
                    .sum();
                p.constant + min_contrib
            })
            .fold(f64::INFINITY, f64::min);

        let ub = vf
            .pieces
            .iter()
            .map(|p| {
                let max_contrib: f64 = p
                    .coefficients
                    .iter()
                    .enumerate()
                    .map(|(i, &a)| {
                        let (lo, hi) = self.x_bound(i, problem);
                        if a >= 0.0 {
                            a * hi
                        } else {
                            a * lo
                        }
                    })
                    .sum();
                p.constant + max_contrib
            })
            .fold(f64::NEG_INFINITY, f64::max);

        let safe_lb = if lb.is_finite() { lb } else { -1e8 };
        let safe_ub = if ub.is_finite() { ub } else { 1e8 };

        let var = MilpVariable {
            name: "phi".to_string(),
            lower_bound: safe_lb,
            upper_bound: safe_ub,
            obj_coeff: 0.0,
            var_type: VarType::Continuous,
        };
        let idx = milp.add_variable(var);
        debug_assert_eq!(idx, layout.phi_var);
    }

    // ── McCormick envelopes ────────────────────────────────────────

    /// Add McCormick auxiliary variables and return the relaxation descriptors.
    pub fn add_mccormick_for_bilinear(
        &self,
        milp: &mut MilpProblem,
        bilinear: &[(usize, usize)],
        layout: &VFVarLayout,
    ) -> Vec<McCormickRelaxation> {
        let mut relaxations = Vec::with_capacity(bilinear.len());
        for (idx, &(xi, yj)) in bilinear.iter().enumerate() {
            let aux_idx = layout.mccormick_start + idx;
            let (x_lo, x_hi) = self.x_bound_raw(xi);
            let y_lo = 0.0;
            let y_hi = 1e6;

            let var = MilpVariable {
                name: format!("mc_{}_{}", xi, yj),
                lower_bound: f64::min(
                    x_lo * y_lo,
                    f64::min(x_lo * y_hi, f64::min(x_hi * y_lo, x_hi * y_hi)),
                ),
                upper_bound: f64::max(
                    x_lo * y_lo,
                    f64::max(x_lo * y_hi, f64::max(x_hi * y_lo, x_hi * y_hi)),
                ),
                obj_coeff: 0.0,
                var_type: VarType::Continuous,
            };
            let actual_idx = milp.add_variable(var);
            debug_assert_eq!(actual_idx, aux_idx);

            relaxations.push(McCormickRelaxation {
                x_var: layout.x_idx(xi),
                y_var: layout.y_idx(yj),
                aux_var: aux_idx,
                x_lo,
                x_hi,
                y_lo,
                y_hi,
            });
        }
        relaxations
    }

    // ── upper-level objective ──────────────────────────────────────

    /// Set the upper-level objective  min c_x^T x + c_y^T y  on the MILP.
    pub fn build_upper_objective(
        &self,
        milp: &mut MilpProblem,
        problem: &BilevelProblem,
        layout: &VFVarLayout,
    ) {
        for (i, &cx) in problem.upper_obj_c_x.iter().enumerate() {
            milp.set_obj_coeff(layout.x_idx(i), cx);
        }
        for (j, &cy) in problem.upper_obj_c_y.iter().enumerate() {
            milp.set_obj_coeff(layout.y_idx(j), cy);
        }
    }

    // ── upper-level constraints ────────────────────────────────────

    /// Emit the upper-level constraints (if any) into the MILP.
    /// The upper-level constraint matrix is stored as a sparse matrix
    /// over both x and y, with rhs in `upper_constraints_b`.
    pub fn add_upper_constraints(
        &self,
        milp: &mut MilpProblem,
        problem: &BilevelProblem,
        layout: &VFVarLayout,
    ) {
        let m = problem.num_upper_constraints;
        for row in 0..m {
            let rhs = if row < problem.upper_constraints_b.len() {
                problem.upper_constraints_b[row]
            } else {
                0.0
            };
            let mut con = MilpConstraint::new(&format!("upper_{}", row), ConstraintSense::Le, rhs);

            // The upper constraint matrix is over the joint (x,y) space.
            // Columns [0, num_upper_vars) correspond to x;
            // Columns [num_upper_vars, num_upper_vars + num_lower_vars) to y.
            for entry in &problem.upper_constraints_a.entries {
                if entry.row == row {
                    if entry.col < problem.num_upper_vars {
                        con.add_term(layout.x_idx(entry.col), entry.value);
                    } else {
                        let y_col = entry.col - problem.num_upper_vars;
                        if y_col < problem.num_lower_vars {
                            con.add_term(layout.y_idx(y_col), entry.value);
                        }
                    }
                }
            }

            milp.add_constraint(con);
        }
    }

    // ── sampling ───────────────────────────────────────────────────

    /// Generate sample points in x-space using a stratified grid with
    /// random perturbation inside each stratum.
    pub fn sample_x_points(&self, problem: &BilevelProblem) -> Vec<Vec<f64>> {
        let n = problem.num_upper_vars;
        let num = self.config.sampling_points.max(1);

        let bounds: Vec<(f64, f64)> = (0..n).map(|i| self.x_bound(i, problem)).collect();

        let mut points = Vec::with_capacity(num);

        // Include the center and corners first
        let center: Vec<f64> = bounds.iter().map(|(lo, hi)| (lo + hi) / 2.0).collect();
        points.push(center);

        // Lower-bound corner
        let lb_corner: Vec<f64> = bounds.iter().map(|(lo, _)| *lo).collect();
        points.push(lb_corner);

        // Upper-bound corner
        let ub_corner: Vec<f64> = bounds.iter().map(|(_, hi)| *hi).collect();
        points.push(ub_corner);

        // Fill remaining with stratified sampling
        let remaining = num.saturating_sub(points.len());
        if remaining > 0 && n > 0 {
            for k in 0..remaining {
                let t = (k as f64 + 0.5) / remaining as f64;
                let point: Vec<f64> = bounds
                    .iter()
                    .enumerate()
                    .map(|(d, &(lo, hi))| {
                        let phase = ((d as f64 + 1.0) * t * std::f64::consts::PI).sin();
                        let frac = 0.5 + 0.5 * phase;
                        lo + frac * (hi - lo)
                    })
                    .collect();
                points.push(point);
            }
        }

        points.truncate(num);
        points
    }

    // ── lower-level oracle ─────────────────────────────────────────

    /// Solve the lower-level LP at a fixed x, returning `(objective, dual)`.
    pub fn solve_lower_level_at(
        &self,
        problem: &BilevelProblem,
        x: &[f64],
    ) -> Option<(f64, Vec<f64>)> {
        let lp = problem.lower_level_lp(x);
        let sol = solve_lp(&lp).ok()?;
        if sol.status != LpStatus::Optimal {
            return None;
        }
        Some((sol.objective, sol.dual.clone()))
    }

    /// Construct an affine piece  α^T x + β  from a dual solution π
    /// and the lower-level data  b, B.
    ///
    /// By LP duality:  φ(x) = π^T (b + Bx)  at the optimal dual π,
    /// so  α = B^T π,  β = π^T b.
    pub fn construct_affine_piece(dual: &[f64], b: &[f64], linking: &SparseMatrix) -> AffinePiece {
        // β = π^T b
        let constant: f64 = dual.iter().zip(b.iter()).map(|(pi, bi)| pi * bi).sum();

        // α = B^T π  (linking is m×n_x, so α has dimension n_x = linking.cols)
        let mut coefficients = vec![0.0; linking.cols];
        for entry in &linking.entries {
            if entry.row < dual.len() {
                coefficients[entry.col] += entry.value * dual[entry.row];
            }
        }

        AffinePiece {
            coefficients,
            constant,
            region: None,
        }
    }

    // ── private helpers ────────────────────────────────────────────

    /// Bounds on x_i, falling back to user config, then to defaults.
    fn x_bound(&self, i: usize, _problem: &BilevelProblem) -> (f64, f64) {
        self.x_bound_raw(i)
    }

    fn x_bound_raw(&self, i: usize) -> (f64, f64) {
        if i < self.config.x_bounds.len() {
            self.config.x_bounds[i]
        } else {
            (0.0, 100.0)
        }
    }

    fn add_upper_variables(
        &self,
        milp: &mut MilpProblem,
        problem: &BilevelProblem,
        layout: &VFVarLayout,
    ) {
        for i in 0..problem.num_upper_vars {
            let (lb, ub) = self.x_bound(i, problem);
            let var = MilpVariable {
                name: format!("x_{}", i),
                lower_bound: lb,
                upper_bound: ub,
                obj_coeff: 0.0,
                var_type: VarType::Continuous,
            };
            let idx = milp.add_variable(var);
            debug_assert_eq!(idx, layout.x_idx(i));
        }
    }

    fn add_lower_variables(
        &self,
        milp: &mut MilpProblem,
        problem: &BilevelProblem,
        layout: &VFVarLayout,
    ) {
        for j in 0..problem.num_lower_vars {
            let var = MilpVariable {
                name: format!("y_{}", j),
                lower_bound: 0.0,
                upper_bound: 1e8,
                obj_coeff: 0.0,
                var_type: VarType::Continuous,
            };
            let idx = milp.add_variable(var);
            debug_assert_eq!(idx, layout.y_idx(j));
        }
    }

    /// Detect bilinear x_i · y_j terms from the upper constraint matrix.
    fn detect_bilinear_terms(&self, problem: &BilevelProblem) -> Vec<(usize, usize)> {
        if !self.config.add_mccormick {
            return Vec::new();
        }
        let mut terms = Vec::new();
        let mut seen = std::collections::HashSet::new();

        for entry in &problem.upper_constraints_a.entries {
            let col = entry.col;
            if col >= problem.num_upper_vars {
                let y_col = col - problem.num_upper_vars;
                // Look for an x term in the same row
                for other in &problem.upper_constraints_a.entries {
                    if other.row == entry.row && other.col < problem.num_upper_vars {
                        let pair = (other.col, y_col);
                        if seen.insert(pair) {
                            terms.push(pair);
                        }
                    }
                }
            }
        }
        terms
    }

    /// Trim PWL pieces to the configured maximum by keeping the most
    /// diverse subset (greedy farthest-point selection on gradient space).
    fn trim_pieces(&self, pwl: &mut PiecewiseLinearVF) {
        let max = self.config.max_pieces;
        if pwl.pieces.len() <= max {
            return;
        }

        let mut kept = Vec::with_capacity(max);
        let mut remaining: Vec<usize> = (0..pwl.pieces.len()).collect();

        // Seed with the first piece
        kept.push(remaining.remove(0));

        while kept.len() < max && !remaining.is_empty() {
            // Pick the piece whose gradient is farthest from all kept gradients
            let mut best_idx = 0;
            let mut best_dist = f64::NEG_INFINITY;

            for (pos, &r) in remaining.iter().enumerate() {
                let min_dist = kept
                    .iter()
                    .map(|&k| gradient_distance(&pwl.pieces[r], &pwl.pieces[k]))
                    .fold(f64::INFINITY, f64::min);

                if min_dist > best_dist {
                    best_dist = min_dist;
                    best_idx = pos;
                }
            }

            kept.push(remaining.remove(best_idx));
        }

        kept.sort_unstable();
        let new_pieces: Vec<AffinePiece> =
            kept.into_iter().map(|i| pwl.pieces[i].clone()).collect();
        pwl.pieces = new_pieces;
    }
}

/// Euclidean distance between gradients of two affine pieces.
fn gradient_distance(a: &AffinePiece, b: &AffinePiece) -> f64 {
    a.coefficients
        .iter()
        .zip(b.coefficients.iter())
        .map(|(ai, bi)| (ai - bi).powi(2))
        .sum::<f64>()
        .sqrt()
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use bicut_types::{BilevelProblem, SparseMatrix};

    /// Build a tiny bilevel problem for testing:
    ///
    /// upper:  min  x + y
    /// lower:  min  y   s.t.  y ≤ 1 + x,  y ≥ 0
    ///
    /// So A = [[1]], b = [1], B = [[1]], c = [1].
    /// φ(x) = min { y : y ≤ 1+x, y ≥ 0 } = 0 for x ≥ −1.
    fn simple_problem() -> BilevelProblem {
        let mut lower_a = SparseMatrix::new(1, 1);
        lower_a.add_entry(0, 0, 1.0); // y ≤ ...

        let mut lower_linking = SparseMatrix::new(1, 1);
        lower_linking.add_entry(0, 0, 1.0); // ... + 1·x

        BilevelProblem {
            upper_obj_c_x: vec![1.0],
            upper_obj_c_y: vec![1.0],
            lower_obj_c: vec![1.0],
            lower_a,
            lower_b: vec![1.0],
            lower_linking_b: lower_linking,
            upper_constraints_a: SparseMatrix::new(0, 2),
            upper_constraints_b: vec![],
            num_upper_vars: 1,
            num_lower_vars: 1,
            num_lower_constraints: 1,
            num_upper_constraints: 0,
        }
    }

    /// A slightly richer problem with 2 x-vars, 2 y-vars, 3 constraints.
    fn two_var_problem() -> BilevelProblem {
        // lower: min y1 + 2*y2
        // s.t. y1 + y2 ≤ 3 + x1
        //      y1      ≤ 2 + x2
        //           y2 ≤ 2
        let mut lower_a = SparseMatrix::new(3, 2);
        lower_a.add_entry(0, 0, 1.0);
        lower_a.add_entry(0, 1, 1.0);
        lower_a.add_entry(1, 0, 1.0);
        lower_a.add_entry(2, 1, 1.0);

        let mut linking = SparseMatrix::new(3, 2);
        linking.add_entry(0, 0, 1.0);
        linking.add_entry(1, 1, 1.0);

        BilevelProblem {
            upper_obj_c_x: vec![1.0, 1.0],
            upper_obj_c_y: vec![1.0, 1.0],
            lower_obj_c: vec![1.0, 2.0],
            lower_a,
            lower_b: vec![3.0, 2.0, 2.0],
            lower_linking_b: linking,
            upper_constraints_a: SparseMatrix::new(0, 4),
            upper_constraints_b: vec![],
            num_upper_vars: 2,
            num_lower_vars: 2,
            num_lower_constraints: 3,
            num_upper_constraints: 0,
        }
    }

    fn default_config() -> ValueFunctionConfig {
        ValueFunctionConfig {
            tolerance: 1e-8,
            max_pieces: 200,
            sampling_points: 10,
            add_mccormick: false,
            mccormick_partitions: 4,
            x_bounds: vec![(0.0, 10.0)],
        }
    }

    fn two_var_config() -> ValueFunctionConfig {
        ValueFunctionConfig {
            tolerance: 1e-8,
            max_pieces: 200,
            sampling_points: 20,
            add_mccormick: false,
            mccormick_partitions: 4,
            x_bounds: vec![(0.0, 10.0), (0.0, 10.0)],
        }
    }

    // ── 1. VFVarLayout ─────────────────────────────────────────────

    #[test]
    fn test_var_layout() {
        let prob = simple_problem();
        let layout = VFVarLayout::new(&prob, 3);

        assert_eq!(layout.x_start, 0);
        assert_eq!(layout.x_count, 1);
        assert_eq!(layout.y_start, 1);
        assert_eq!(layout.y_count, 1);
        assert_eq!(layout.phi_var, 2);
        assert_eq!(layout.mccormick_start, 3);
        assert_eq!(layout.mccormick_count, 3);
        assert_eq!(layout.total_vars, 6);
        assert_eq!(layout.x_idx(0), 0);
        assert_eq!(layout.y_idx(0), 1);

        // Two-variable problem
        let prob2 = two_var_problem();
        let layout2 = VFVarLayout::new(&prob2, 0);
        assert_eq!(layout2.x_count, 2);
        assert_eq!(layout2.y_count, 2);
        assert_eq!(layout2.phi_var, 4);
        assert_eq!(layout2.total_vars, 5);
        assert_eq!(layout2.x_idx(1), 1);
        assert_eq!(layout2.y_idx(1), 3);
    }

    // ── 2. Value function computation ──────────────────────────────

    #[test]
    fn test_value_function_computation() {
        let prob = simple_problem();
        let cfg = default_config();
        let pass = ValueFunctionPass::new(cfg);

        let vf = pass.compute_value_function(&prob).unwrap();
        assert!(vf.num_pieces() > 0);

        // For the simple problem, φ(x) = 0 when x ≥ -1.
        // Since x ∈ [0, 10], the value function is identically 0.
        for x_val in [0.0, 1.0, 5.0, 10.0] {
            let v = vf.evaluate(&[x_val]);
            assert!(v.abs() < 1.0, "φ({}) = {} should be near 0", x_val, v,);
        }
    }

    // ── 3. Epigraph formulation ────────────────────────────────────

    #[test]
    fn test_epigraph_variable() {
        let prob = simple_problem();
        let cfg = default_config();
        let pass = ValueFunctionPass::new(cfg);
        let vf = pass.compute_value_function(&prob).unwrap();
        let layout = VFVarLayout::new(&prob, 0);

        let mut milp = MilpProblem::new("test_epi");
        // Add placeholder x and y variables first
        milp.add_variable(MilpVariable::continuous("x_0", 0.0, 10.0));
        milp.add_variable(MilpVariable::continuous("y_0", 0.0, 1e8));

        pass.add_epigraph_variable(&mut milp, &prob, &vf, &layout);

        assert_eq!(milp.num_vars(), 3);
        let phi = &milp.variables[layout.phi_var];
        assert_eq!(phi.name, "phi");
        assert_eq!(phi.var_type, VarType::Continuous);
        assert!(phi.lower_bound <= phi.upper_bound);
    }

    // ── 4. PWL constraint encoding ─────────────────────────────────

    #[test]
    fn test_pwl_constraint_encoding() {
        let prob = simple_problem();
        let cfg = default_config();
        let pass = ValueFunctionPass::new(cfg);
        let vf = pass.compute_value_function(&prob).unwrap();
        let num_pieces = vf.num_pieces();
        let layout = VFVarLayout::new(&prob, 0);

        let mut milp = MilpProblem::new("test_pwl");
        milp.add_variable(MilpVariable::continuous("x_0", 0.0, 10.0));
        milp.add_variable(MilpVariable::continuous("y_0", 0.0, 1e8));
        milp.add_variable(MilpVariable::continuous("phi", -1e8, 1e8));

        pass.add_value_function_constraint(&mut milp, &prob, &vf, &layout);

        // Should have 1 optimality cut + num_pieces epigraph constraints
        assert_eq!(milp.num_constraints(), 1 + num_pieces);

        // The first constraint should be the optimality cut: c^T y - t ≤ 0
        let opt = &milp.constraints[0];
        assert_eq!(opt.name, "vf_optimality");
        assert_eq!(opt.sense, ConstraintSense::Le);

        // Each remaining constraint names should be vf_piece_*
        for k in 0..num_pieces {
            let con = &milp.constraints[1 + k];
            assert_eq!(con.name, format!("vf_piece_{}", k));
            assert_eq!(con.sense, ConstraintSense::Ge);
        }
    }

    // ── 5. McCormick relaxation ────────────────────────────────────

    #[test]
    fn test_mccormick_relaxation() {
        let mc = McCormickRelaxation {
            x_var: 0,
            y_var: 1,
            aux_var: 2,
            x_lo: 0.0,
            x_hi: 1.0,
            y_lo: 0.0,
            y_hi: 1.0,
        };

        let constraints = mc.constraints();
        assert_eq!(constraints.len(), 4);

        // Verify constraint senses: first 2 are Ge, last 2 are Le
        assert_eq!(constraints[0].1, ConstraintSense::Ge);
        assert_eq!(constraints[1].1, ConstraintSense::Ge);
        assert_eq!(constraints[2].1, ConstraintSense::Le);
        assert_eq!(constraints[3].1, ConstraintSense::Le);

        // For x,y ∈ [0,1], check that w = x*y satisfies all constraints
        // at the interior point x=0.5, y=0.5, w=0.25
        let x_val = 0.5;
        let y_val = 0.5;
        let w_val = 0.25;
        let vals = [x_val, y_val, w_val];

        for (coeffs, sense, rhs) in &constraints {
            let lhs: f64 = coeffs.iter().map(|&(var, coeff)| coeff * vals[var]).sum();
            match sense {
                ConstraintSense::Ge => {
                    assert!(lhs >= rhs - 1e-10, "Ge violated: {} < {}", lhs, rhs,);
                }
                ConstraintSense::Le => {
                    assert!(lhs <= rhs + 1e-10, "Le violated: {} > {}", lhs, rhs,);
                }
                ConstraintSense::Eq => {
                    assert!((lhs - rhs).abs() < 1e-10, "Eq violated");
                }
            }
        }

        // Also check at vertices (0,0), (1,1), (0,1), (1,0)
        let vertices = [
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            (0.0, 1.0, 0.0),
            (1.0, 0.0, 0.0),
        ];
        for (xv, yv, wv) in &vertices {
            let v = [*xv, *yv, *wv];
            for (coeffs, sense, rhs) in &constraints {
                let lhs: f64 = coeffs.iter().map(|&(var, coeff)| coeff * v[var]).sum();
                match sense {
                    ConstraintSense::Ge => assert!(lhs >= rhs - 1e-10),
                    ConstraintSense::Le => assert!(lhs <= rhs + 1e-10),
                    ConstraintSense::Eq => assert!((lhs - rhs).abs() < 1e-10),
                }
            }
        }
    }

    // ── 6. Sample point generation ─────────────────────────────────

    #[test]
    fn test_sample_points() {
        let prob = two_var_problem();
        let cfg = two_var_config();
        let pass = ValueFunctionPass::new(cfg.clone());

        let points = pass.sample_x_points(&prob);
        assert_eq!(points.len(), cfg.sampling_points);

        // All points should be within bounds
        for pt in &points {
            assert_eq!(pt.len(), 2);
            for (d, &v) in pt.iter().enumerate() {
                let (lo, hi) = cfg.x_bounds[d];
                assert!(
                    v >= lo - 1e-10 && v <= hi + 1e-10,
                    "x[{}] = {} out of [{}, {}]",
                    d,
                    v,
                    lo,
                    hi,
                );
            }
        }

        // First point is the center
        let center = &points[0];
        assert!((center[0] - 5.0).abs() < 1e-10);
        assert!((center[1] - 5.0).abs() < 1e-10);
    }

    // ── 7. Affine piece construction ───────────────────────────────

    #[test]
    fn test_construct_affine_piece() {
        // π = [2.0, 3.0],  b = [1.0, 4.0],  B = [[1, 0], [0, 2]] (2 constraints × 2 x-vars)
        let dual = vec![2.0, 3.0];
        let b = vec![1.0, 4.0];
        let mut linking = SparseMatrix::new(2, 2);
        linking.add_entry(0, 0, 1.0);
        linking.add_entry(1, 1, 2.0);

        let piece = ValueFunctionPass::construct_affine_piece(&dual, &b, &linking);

        // β = π^T b = 2*1 + 3*4 = 14
        assert!(
            (piece.constant - 14.0).abs() < 1e-10,
            "β = {}",
            piece.constant
        );

        // α = B^T π:
        //   α[0] = B[0,0]*π[0] + B[1,0]*π[1] = 1*2 + 0*3 = 2
        //   α[1] = B[0,1]*π[0] + B[1,1]*π[1] = 0*2 + 2*3 = 6
        assert_eq!(piece.coefficients.len(), 2);
        assert!(
            (piece.coefficients[0] - 2.0).abs() < 1e-10,
            "α[0] = {}",
            piece.coefficients[0]
        );
        assert!(
            (piece.coefficients[1] - 6.0).abs() < 1e-10,
            "α[1] = {}",
            piece.coefficients[1]
        );

        // Evaluate at x = [1, 1]: piece(x) = 2*1 + 6*1 + 14 = 22
        assert!((piece.evaluate(&[1.0, 1.0]) - 22.0).abs() < 1e-10);
    }

    // ── 8. Full apply ──────────────────────────────────────────────

    #[test]
    fn test_full_apply() {
        let prob = simple_problem();
        let cfg = default_config();
        let pass = ValueFunctionPass::new(cfg);

        let result = pass.apply(&prob).unwrap();

        // Variables: 1 x + 1 y + 1 phi = 3
        assert_eq!(result.milp.num_vars(), 3);
        assert_eq!(result.num_mccormick_vars, 0);
        assert!(result.num_pieces > 0);
        assert!(result.milp.num_constraints() > 0);

        // Objective is min x + y
        assert!((result.milp.variables[0].obj_coeff - 1.0).abs() < 1e-10);
        assert!((result.milp.variables[1].obj_coeff - 1.0).abs() < 1e-10);
        assert!((result.milp.variables[2].obj_coeff).abs() < 1e-10);

        // Expected constraints:
        //   1 primal feasibility  (y ≤ 1 + x  ⟹  y - x ≤ 1)
        // + 1 optimality cut      (y ≤ t)
        // + num_pieces epigraph constraints
        let expected_constraints = 1 + 1 + result.num_pieces;
        assert_eq!(result.milp.num_constraints(), expected_constraints);

        // Check that the MILP sense is Minimize
        assert_eq!(result.milp.sense, OptDirection::Minimize);

        // Value function should be available
        assert!(result.value_function.num_pieces() > 0);

        // Warnings should be empty for this small problem
        assert!(
            result.warnings.is_empty(),
            "unexpected warnings: {:?}",
            result.warnings,
        );

        // Verify the two-variable problem also works
        let prob2 = two_var_problem();
        let cfg2 = two_var_config();
        let pass2 = ValueFunctionPass::new(cfg2);
        let result2 = pass2.apply(&prob2).unwrap();

        // 2 x + 2 y + 1 phi = 5
        assert_eq!(result2.milp.num_vars(), 5);
        assert!(result2.milp.num_constraints() > 0);
        assert!(result2.num_pieces > 0);
    }
}
