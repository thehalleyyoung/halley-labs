//! KKT (Karush–Kuhn–Tucker) reformulation pass for bilevel optimisation.
//!
//! Given a bilevel program of the form
//!
//! ```text
//!   min_{x,y}  F(x,y)        (upper-level objective)
//!   s.t.       G(x,y) ≤ 0    (upper-level constraints)
//!              y ∈ argmin_{y'} { c^T y' : Ay' ≤ b + Bx, y' ≥ 0 }
//! ```
//!
//! this pass replaces the lower-level optimality condition with the KKT
//! system:
//!
//! 1. **Primal feasibility** — `Ay ≤ b + Bx`,  `y ≥ 0`
//! 2. **Dual feasibility**   — `A^T λ ≥ c`,  `λ ≥ 0`  (introducing dual
//!    variables λ)
//! 3. **Stationarity**       — `A^T λ − s = c`  (with slack `s ≥ 0`; for
//!    the LP case this is equivalent to dual feasibility)
//! 4. **Complementarity**    — `λ_i · (b_i + (Bx)_i − (Ay)_i) = 0`
//!    *and* `s_j · y_j = 0`
//!
//! The bilinear complementarity products are linearised using one of three
//! encodings selected by [`ComplementarityEncoding`]:
//!
//! | Encoding | Mechanism |
//! |----------|-----------|
//! | **Big-M** | `λ_i ≤ M z_i`, `slack_i ≤ M' (1−z_i)`, `z_i ∈ {0,1}` |
//! | **SOS1** | `(λ_i, slack_i)` in an SOS-1 set |
//! | **Indicator** | `z_i=1 → λ_i=0`, `z_i=0 → slack_i=0` |

use crate::bigm::{BigMComputer, BigMConfig, BigMResult, BigMSet};
use crate::pipeline::{
    IndicatorConstraint, MilpConstraint, MilpProblem, MilpVariable, Sos1Set, VarType,
};
use crate::{CompilerConfig, CompilerError, CompilerResult, ComplementarityEncoding};

use bicut_types::{
    BilevelProblem, ConstraintSense, CqStatus, OptDirection, SparseMatrix, DEFAULT_TOLERANCE,
};

// ---------------------------------------------------------------------------
// KktConfig
// ---------------------------------------------------------------------------

/// Per-pass configuration for the KKT reformulation.
#[derive(Debug, Clone)]
pub struct KktConfig {
    /// How complementarity constraints are linearised.
    pub encoding: ComplementarityEncoding,
    /// Big-M computation settings (used only when `encoding == BigM`).
    pub bigm_config: BigMConfig,
    /// Whether to verify a constraint qualification before reformulation.
    pub verify_cq: bool,
    /// Numerical tolerance for zero comparisons.
    pub tolerance: f64,
    /// If `true`, add tightening cuts derived from variable bounds.
    pub add_bound_cuts: bool,
}

impl Default for KktConfig {
    fn default() -> Self {
        Self {
            encoding: ComplementarityEncoding::BigM,
            bigm_config: BigMConfig::default(),
            verify_cq: true,
            tolerance: DEFAULT_TOLERANCE,
            add_bound_cuts: false,
        }
    }
}

impl KktConfig {
    /// Build a [`KktConfig`] from the global [`CompilerConfig`].
    pub fn from_compiler_config(cc: &CompilerConfig) -> Self {
        Self {
            encoding: cc.complementarity_encoding,
            bigm_config: BigMConfig::default(),
            verify_cq: true,
            tolerance: cc.tolerance,
            add_bound_cuts: false,
        }
    }
}

// ---------------------------------------------------------------------------
// KktPassResult
// ---------------------------------------------------------------------------

/// Output of the KKT reformulation pass.
#[derive(Debug)]
pub struct KktPassResult {
    /// The reformulated single-level MILP.
    pub milp: MilpProblem,
    /// Big-M values used, if Big-M encoding was selected.
    pub bigm_set: Option<BigMSet>,
    /// Constraint-qualification status of the lower level.
    pub cq_status: CqStatus,
    /// Number of binary variables introduced for complementarity.
    pub num_binary_vars: usize,
    /// Number of dual variables (λ) introduced.
    pub num_dual_vars: usize,
    /// Total number of complementarity pairs linearised.
    pub num_complementarity_pairs: usize,
    /// Non-fatal warnings emitted during the pass.
    pub warnings: Vec<String>,
}

/// Alias used by the top-level re-export in `lib.rs`.
pub type KktReformulation = KktPassResult;

// ---------------------------------------------------------------------------
// VarLayout — index bookkeeping for the reformulated MILP
// ---------------------------------------------------------------------------

/// Tracks where each class of variable lives inside the flat MILP variable
/// vector.
#[derive(Debug, Clone)]
pub struct VarLayout {
    pub x_start: usize,
    pub x_count: usize,
    pub y_start: usize,
    pub y_count: usize,
    pub lambda_start: usize,
    pub lambda_count: usize,
    pub primal_slack_start: usize,
    pub primal_slack_count: usize,
    pub dual_slack_start: usize,
    pub dual_slack_count: usize,
    pub binary_start: usize,
    pub binary_count: usize,
    pub total_vars: usize,
}

impl VarLayout {
    /// Compute the variable layout from a [`BilevelProblem`] and encoding.
    ///
    /// Binary variables are allocated only for Big-M and Indicator encodings;
    /// SOS1 does not require them.
    pub fn new(problem: &BilevelProblem, encoding: ComplementarityEncoding) -> Self {
        let nx = problem.num_upper_vars;
        let ny = problem.num_lower_vars;
        let m = problem.num_lower_constraints;

        let x_start = 0;
        let x_count = nx;

        let y_start = x_start + x_count;
        let y_count = ny;

        let lambda_start = y_start + y_count;
        let lambda_count = m;

        let primal_slack_start = lambda_start + lambda_count;
        let primal_slack_count = m;

        let dual_slack_start = primal_slack_start + primal_slack_count;
        let dual_slack_count = ny;

        let binary_start = dual_slack_start + dual_slack_count;
        let num_complementarity = m + ny;
        let binary_count = match encoding {
            ComplementarityEncoding::BigM | ComplementarityEncoding::Indicator => {
                num_complementarity
            }
            ComplementarityEncoding::SOS1 => 0,
        };

        let total_vars = binary_start + binary_count;

        Self {
            x_start,
            x_count,
            y_start,
            y_count,
            lambda_start,
            lambda_count,
            primal_slack_start,
            primal_slack_count,
            dual_slack_start,
            dual_slack_count,
            binary_start,
            binary_count,
            total_vars,
        }
    }

    /// MILP index of the `i`-th upper-level (x) variable.
    #[inline]
    pub fn x_idx(&self, i: usize) -> usize {
        self.x_start + i
    }

    /// MILP index of the `j`-th lower-level (y) variable.
    #[inline]
    pub fn y_idx(&self, j: usize) -> usize {
        self.y_start + j
    }

    /// MILP index of the `i`-th dual variable (λ_i).
    #[inline]
    pub fn lambda_idx(&self, i: usize) -> usize {
        self.lambda_start + i
    }

    /// MILP index of the `i`-th primal slack variable.
    #[inline]
    pub fn primal_slack_idx(&self, i: usize) -> usize {
        self.primal_slack_start + i
    }

    /// MILP index of the `j`-th dual slack variable.
    #[inline]
    pub fn dual_slack_idx(&self, j: usize) -> usize {
        self.dual_slack_start + j
    }

    /// MILP index of the `k`-th binary indicator variable.
    #[inline]
    pub fn binary_idx(&self, k: usize) -> usize {
        self.binary_start + k
    }
}

// ---------------------------------------------------------------------------
// KktPass
// ---------------------------------------------------------------------------

/// The KKT reformulation pass.
///
/// Construct with [`KktPass::new`], then call [`KktPass::apply`] to
/// transform a [`BilevelProblem`] into a single-level [`MilpProblem`].
pub struct KktPass {
    config: KktConfig,
}

impl KktPass {
    /// Create a new KKT pass with the given configuration.
    pub fn new(config: KktConfig) -> Self {
        Self { config }
    }

    // ─── main entry point ────────────────────────────────────────────

    /// Apply the full KKT reformulation to `problem`, returning a
    /// [`KktPassResult`] containing the reformulated MILP plus metadata.
    pub fn apply(&self, problem: &BilevelProblem) -> Result<KktPassResult, CompilerError> {
        // 1. Validate the problem dimensions.
        self.validate_problem(problem)?;

        // 2. Compute variable layout.
        let layout = VarLayout::new(problem, self.config.encoding);

        // 3. Initialise an empty MILP with the right number of variables.
        let mut milp = MilpProblem::new("kkt_reformulation");
        milp.sense = OptDirection::Minimize;
        self.allocate_variables(problem, &mut milp, &layout);

        // 4. Upper-level objective.
        self.build_upper_objective(problem, &mut milp, &layout);

        // 5. Upper-level constraints  G(x,y) ≤ 0.
        self.add_upper_constraints(problem, &mut milp, &layout);

        // 6. Primal feasibility of the lower level.
        self.add_primal_feasibility(problem, &mut milp, &layout);

        // 7. Dual variables & dual feasibility (stationarity).
        self.add_dual_variables(problem, &mut milp, &layout);
        self.add_dual_feasibility(problem, &mut milp, &layout);

        // 8. Complementarity constraints.
        let mut bigm_set: Option<BigMSet> = None;
        let mut warnings = Vec::new();

        match self.config.encoding {
            ComplementarityEncoding::BigM => {
                let computer = BigMComputer::new(self.config.bigm_config.clone());
                let bm = computer.compute_all_bigms(problem);
                if !bm.all_finite {
                    warnings.push(
                        "Some big-M values are not finite; reformulation may be weak.".into(),
                    );
                }
                self.add_complementarity_bigm(problem, &mut milp, &layout, &bm);
                bigm_set = Some(bm);
            }
            ComplementarityEncoding::SOS1 => {
                self.add_complementarity_sos1(problem, &mut milp, &layout);
            }
            ComplementarityEncoding::Indicator => {
                self.add_complementarity_indicator(problem, &mut milp, &layout);
            }
        }

        // 9. Optional bound-tightening cuts.
        if self.config.add_bound_cuts {
            self.add_bound_cuts(problem, &mut milp, &layout);
        }

        // 10. Constraint qualification check.
        let cq_status = if self.config.verify_cq {
            let st = self.check_constraint_qualification(problem);
            if st == CqStatus::Violated {
                warnings
                    .push("LICQ appears violated; KKT conditions may not be sufficient.".into());
            }
            st
        } else {
            CqStatus::Unknown
        };

        // 11. Assemble result.
        let comp_pairs = Self::compute_complementarity_count(problem);

        Ok(KktPassResult {
            milp,
            bigm_set,
            cq_status,
            num_binary_vars: layout.binary_count,
            num_dual_vars: layout.lambda_count,
            num_complementarity_pairs: comp_pairs,
            warnings,
        })
    }

    // ─── validation ──────────────────────────────────────────────────

    fn validate_problem(&self, p: &BilevelProblem) -> Result<(), CompilerError> {
        if p.num_lower_vars == 0 {
            return Err(CompilerError::Validation(
                "Lower level must have at least one variable.".into(),
            ));
        }
        if p.lower_a.cols != p.num_lower_vars {
            return Err(CompilerError::Validation(format!(
                "lower_a column count ({}) does not match num_lower_vars ({})",
                p.lower_a.cols, p.num_lower_vars,
            )));
        }
        if p.lower_a.rows != p.num_lower_constraints {
            return Err(CompilerError::Validation(format!(
                "lower_a row count ({}) does not match num_lower_constraints ({})",
                p.lower_a.rows, p.num_lower_constraints,
            )));
        }
        if p.lower_obj_c.len() != p.num_lower_vars {
            return Err(CompilerError::Validation(format!(
                "lower_obj_c length ({}) does not match num_lower_vars ({})",
                p.lower_obj_c.len(),
                p.num_lower_vars,
            )));
        }
        if p.lower_b.len() != p.num_lower_constraints {
            return Err(CompilerError::Validation(format!(
                "lower_b length ({}) does not match num_lower_constraints ({})",
                p.lower_b.len(),
                p.num_lower_constraints,
            )));
        }
        if p.upper_obj_c_x.len() != p.num_upper_vars {
            return Err(CompilerError::Validation(format!(
                "upper_obj_c_x length ({}) does not match num_upper_vars ({})",
                p.upper_obj_c_x.len(),
                p.num_upper_vars,
            )));
        }
        if p.upper_obj_c_y.len() != p.num_lower_vars {
            return Err(CompilerError::Validation(format!(
                "upper_obj_c_y length ({}) does not match num_lower_vars ({})",
                p.upper_obj_c_y.len(),
                p.num_lower_vars,
            )));
        }
        Ok(())
    }

    // ─── variable allocation ─────────────────────────────────────────

    fn allocate_variables(
        &self,
        problem: &BilevelProblem,
        milp: &mut MilpProblem,
        layout: &VarLayout,
    ) {
        // x variables (upper-level leader decisions).
        for i in 0..layout.x_count {
            milp.add_variable(MilpVariable::continuous(
                &format!("x_{}", i),
                f64::NEG_INFINITY,
                f64::INFINITY,
            ));
        }

        // y variables (lower-level follower decisions, y ≥ 0).
        for j in 0..layout.y_count {
            milp.add_variable(MilpVariable::continuous(
                &format!("y_{}", j),
                0.0,
                f64::INFINITY,
            ));
        }

        // λ variables (dual multipliers for lower constraints, λ ≥ 0).
        for i in 0..layout.lambda_count {
            milp.add_variable(MilpVariable::continuous(
                &format!("lambda_{}", i),
                0.0,
                f64::INFINITY,
            ));
        }

        // Primal slack variables (b + Bx − Ay, one per lower constraint).
        for i in 0..layout.primal_slack_count {
            milp.add_variable(MilpVariable::continuous(
                &format!("ps_{}", i),
                0.0,
                f64::INFINITY,
            ));
        }

        // Dual slack variables (A^T λ − c, one per lower variable).
        for j in 0..layout.dual_slack_count {
            milp.add_variable(MilpVariable::continuous(
                &format!("ds_{}", j),
                0.0,
                f64::INFINITY,
            ));
        }

        // Binary indicator variables (for Big-M / Indicator encodings).
        for k in 0..layout.binary_count {
            milp.add_variable(MilpVariable::binary(&format!("z_{}", k)));
        }
    }

    // ─── upper-level objective ───────────────────────────────────────

    /// Set the objective:  min  c_x^T x  +  c_y^T y.
    pub fn build_upper_objective(
        &self,
        problem: &BilevelProblem,
        milp: &mut MilpProblem,
        layout: &VarLayout,
    ) {
        for (i, &coeff) in problem.upper_obj_c_x.iter().enumerate() {
            if coeff.abs() > self.config.tolerance {
                milp.set_obj_coeff(layout.x_idx(i), coeff);
            }
        }
        for (j, &coeff) in problem.upper_obj_c_y.iter().enumerate() {
            if coeff.abs() > self.config.tolerance {
                milp.set_obj_coeff(layout.y_idx(j), coeff);
            }
        }
    }

    // ─── upper-level constraints ─────────────────────────────────────

    /// Add upper-level constraints  G(x,y) ≤ 0, stored in the bilevel
    /// problem as `upper_constraints_a · [x; y] ≤ upper_constraints_b`.
    pub fn add_upper_constraints(
        &self,
        problem: &BilevelProblem,
        milp: &mut MilpProblem,
        layout: &VarLayout,
    ) {
        let m_upper = problem.num_upper_constraints;
        if m_upper == 0 {
            return;
        }

        for i in 0..m_upper {
            let mut con = MilpConstraint::new(
                &format!("upper_{}", i),
                ConstraintSense::Le,
                problem.upper_constraints_b[i],
            );

            for entry in &problem.upper_constraints_a.entries {
                if entry.row != i {
                    continue;
                }
                let col = entry.col;
                if col < problem.num_upper_vars {
                    con.add_term(layout.x_idx(col), entry.value);
                } else {
                    let j = col - problem.num_upper_vars;
                    if j < problem.num_lower_vars {
                        con.add_term(layout.y_idx(j), entry.value);
                    }
                }
            }

            milp.add_constraint(con);
        }
    }

    // ─── primal feasibility ──────────────────────────────────────────

    /// Add the lower-level primal constraints with explicit slacks:
    ///
    /// ```text
    ///   (Ay)_i + ps_i = b_i + (Bx)_i     for each lower constraint i
    ///   ps_i ≥ 0                          (enforced by variable bounds)
    /// ```
    ///
    /// This is equivalent to `Ay ≤ b + Bx` with `ps = b + Bx − Ay`.
    pub fn add_primal_feasibility(
        &self,
        problem: &BilevelProblem,
        milp: &mut MilpProblem,
        layout: &VarLayout,
    ) {
        let m = problem.num_lower_constraints;

        // Build a row-indexed lookup for lower_a entries.
        let mut a_rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); m];
        for entry in &problem.lower_a.entries {
            if entry.row < m {
                a_rows[entry.row].push((entry.col, entry.value));
            }
        }

        // Build a row-indexed lookup for lower_linking_b entries (Bx term).
        let mut b_rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); m];
        for entry in &problem.lower_linking_b.entries {
            if entry.row < m {
                b_rows[entry.row].push((entry.col, entry.value));
            }
        }

        for i in 0..m {
            // (Ay)_i + ps_i − (Bx)_i = b_i
            let mut con = MilpConstraint::new(
                &format!("primal_feas_{}", i),
                ConstraintSense::Eq,
                problem.lower_b[i],
            );

            // +Ay terms.
            for &(col, val) in &a_rows[i] {
                con.add_term(layout.y_idx(col), val);
            }

            // +ps_i (primal slack).
            con.add_term(layout.primal_slack_idx(i), 1.0);

            // −Bx terms (move to LHS: subtract).
            for &(col, val) in &b_rows[i] {
                con.add_term(layout.x_idx(col), -val);
            }

            milp.add_constraint(con);
        }
    }

    // ─── dual variables ──────────────────────────────────────────────

    /// (No-op in terms of constraints; dual variable bounds are set during
    /// allocation.  This hook exists for symmetry and logging.)
    pub fn add_dual_variables(
        &self,
        _problem: &BilevelProblem,
        _milp: &mut MilpProblem,
        _layout: &VarLayout,
    ) {
        // λ ≥ 0 is enforced by variable lower bounds set in allocate_variables.
    }

    // ─── dual feasibility / stationarity ─────────────────────────────

    /// Add stationarity constraints with explicit dual slacks:
    ///
    /// ```text
    ///   (A^T λ)_j − ds_j = c_j    for each lower variable j
    ///   ds_j ≥ 0                   (enforced by variable bounds)
    /// ```
    ///
    /// Equivalently:  `A^T λ ≥ c`  with  `ds = A^T λ − c`.
    pub fn add_dual_feasibility(
        &self,
        problem: &BilevelProblem,
        milp: &mut MilpProblem,
        layout: &VarLayout,
    ) {
        let n = problem.num_lower_vars;

        // Build a column-indexed lookup for lower_a (transpose access).
        let mut a_cols: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        for entry in &problem.lower_a.entries {
            if entry.col < n {
                a_cols[entry.col].push((entry.row, entry.value));
            }
        }

        for j in 0..n {
            // (A^T λ)_j − ds_j = c_j
            let mut con = MilpConstraint::new(
                &format!("stationarity_{}", j),
                ConstraintSense::Eq,
                problem.lower_obj_c[j],
            );

            // +(A^T λ)_j = Σ_i A_{ij} λ_i
            for &(row, val) in &a_cols[j] {
                con.add_term(layout.lambda_idx(row), val);
            }

            // −ds_j
            con.add_term(layout.dual_slack_idx(j), -1.0);

            milp.add_constraint(con);
        }
    }

    // ─── complementarity: Big-M ──────────────────────────────────────

    /// Linearise complementarity via big-M:
    ///
    /// **Constraint complementarity** (i = 0 … m−1):
    /// ```text
    ///   λ_i       ≤  M_dual_i   · z_i
    ///   ps_i      ≤  M_primal_i · (1 − z_i)
    /// ```
    ///
    /// **Variable-bound complementarity** (j = 0 … n−1):
    /// ```text
    ///   ds_j      ≤  M_ds_j     · w_j
    ///   y_j       ≤  M_y_j      · (1 − w_j)
    /// ```
    pub fn add_complementarity_bigm(
        &self,
        problem: &BilevelProblem,
        milp: &mut MilpProblem,
        layout: &VarLayout,
        bigm_set: &BigMSet,
    ) {
        let m = problem.num_lower_constraints;
        let n = problem.num_lower_vars;

        // ── Constraint complementarity ──
        for i in 0..m {
            let z = layout.binary_idx(i);
            let dual_m = bigm_set
                .get_dual_m(i)
                .map(|r| r.dual_bigm)
                .unwrap_or(self.config.bigm_config.default_m);
            let primal_m = bigm_set
                .get_primal_m(i)
                .map(|r| r.primal_bigm)
                .unwrap_or(self.config.bigm_config.default_m);

            // λ_i ≤ M_dual · z_i   ⟹   λ_i − M_dual · z_i ≤ 0
            let mut c1 =
                MilpConstraint::new(&format!("comp_dual_bigm_{}", i), ConstraintSense::Le, 0.0);
            c1.add_term(layout.lambda_idx(i), 1.0);
            c1.add_term(z, -dual_m);
            milp.add_constraint(c1);

            // ps_i ≤ M_primal · (1 − z_i)  ⟹  ps_i + M_primal · z_i ≤ M_primal
            let mut c2 = MilpConstraint::new(
                &format!("comp_primal_bigm_{}", i),
                ConstraintSense::Le,
                primal_m,
            );
            c2.add_term(layout.primal_slack_idx(i), 1.0);
            c2.add_term(z, primal_m);
            milp.add_constraint(c2);
        }

        // ── Variable-bound complementarity ──
        let default_m = self.config.bigm_config.default_m;
        for j in 0..n {
            let w = layout.binary_idx(m + j);

            // ds_j ≤ M · w_j   ⟹   ds_j − M · w_j ≤ 0
            let mut c1 =
                MilpConstraint::new(&format!("comp_ds_bigm_{}", j), ConstraintSense::Le, 0.0);
            c1.add_term(layout.dual_slack_idx(j), 1.0);
            c1.add_term(w, -default_m);
            milp.add_constraint(c1);

            // y_j ≤ M · (1 − w_j)  ⟹  y_j + M · w_j ≤ M
            let mut c2 = MilpConstraint::new(
                &format!("comp_y_bigm_{}", j),
                ConstraintSense::Le,
                default_m,
            );
            c2.add_term(layout.y_idx(j), 1.0);
            c2.add_term(w, default_m);
            milp.add_constraint(c2);
        }
    }

    // ─── complementarity: SOS1 ───────────────────────────────────────

    /// Encode complementarity as SOS-1 sets: in each pair, at most one
    /// member can be non-zero.
    pub fn add_complementarity_sos1(
        &self,
        problem: &BilevelProblem,
        milp: &mut MilpProblem,
        layout: &VarLayout,
    ) {
        let m = problem.num_lower_constraints;
        let n = problem.num_lower_vars;

        // Constraint complementarity: {λ_i, ps_i} in SOS1.
        for i in 0..m {
            let sos = Sos1Set {
                name: format!("sos1_constr_{}", i),
                sos_type: 1,
                members: vec![layout.lambda_idx(i), layout.primal_slack_idx(i)],
                weights: vec![1.0, 2.0],
            };
            milp.add_sos1_set(sos);
        }

        // Variable-bound complementarity: {ds_j, y_j} in SOS1.
        for j in 0..n {
            let sos = Sos1Set {
                name: format!("sos1_var_{}", j),
                sos_type: 1,
                members: vec![layout.dual_slack_idx(j), layout.y_idx(j)],
                weights: vec![1.0, 2.0],
            };
            milp.add_sos1_set(sos);
        }
    }

    // ─── complementarity: Indicator ──────────────────────────────────

    /// Encode complementarity using indicator constraints:
    ///
    /// **Constraint complementarity** (i = 0 … m−1):
    /// ```text
    ///   z_i = 1  →  λ_i  = 0
    ///   z_i = 0  →  ps_i = 0
    /// ```
    ///
    /// **Variable-bound complementarity** (j = 0 … n−1):
    /// ```text
    ///   w_j = 1  →  ds_j = 0
    ///   w_j = 0  →  y_j  = 0
    /// ```
    pub fn add_complementarity_indicator(
        &self,
        problem: &BilevelProblem,
        milp: &mut MilpProblem,
        layout: &VarLayout,
    ) {
        let m = problem.num_lower_constraints;
        let n = problem.num_lower_vars;

        for i in 0..m {
            let z = layout.binary_idx(i);

            // z_i = 1  →  λ_i = 0
            milp.add_indicator_constraint(IndicatorConstraint {
                name: format!("ind_dual_{}", i),
                binary_var: z,
                active_value: true,
                coeffs: vec![(layout.lambda_idx(i), 1.0)],
                sense: ConstraintSense::Eq,
                rhs: 0.0,
            });

            // z_i = 0  →  ps_i = 0
            milp.add_indicator_constraint(IndicatorConstraint {
                name: format!("ind_ps_{}", i),
                binary_var: z,
                active_value: false,
                coeffs: vec![(layout.primal_slack_idx(i), 1.0)],
                sense: ConstraintSense::Eq,
                rhs: 0.0,
            });
        }

        for j in 0..n {
            let w = layout.binary_idx(m + j);

            // w_j = 1  →  ds_j = 0
            milp.add_indicator_constraint(IndicatorConstraint {
                name: format!("ind_ds_{}", j),
                binary_var: w,
                active_value: true,
                coeffs: vec![(layout.dual_slack_idx(j), 1.0)],
                sense: ConstraintSense::Eq,
                rhs: 0.0,
            });

            // w_j = 0  →  y_j = 0
            milp.add_indicator_constraint(IndicatorConstraint {
                name: format!("ind_y_{}", j),
                binary_var: w,
                active_value: false,
                coeffs: vec![(layout.y_idx(j), 1.0)],
                sense: ConstraintSense::Eq,
                rhs: 0.0,
            });
        }
    }

    // ─── optional bound-tightening cuts ──────────────────────────────

    /// Add simple cuts derived from variable bounds and problem structure.
    ///
    /// For each lower constraint `i` we add the implied bound:
    ///   `λ_i + ps_i ≤ M_i`
    /// which couples the dual and primal slack, tightening the LP relaxation.
    fn add_bound_cuts(&self, problem: &BilevelProblem, milp: &mut MilpProblem, layout: &VarLayout) {
        let m = problem.num_lower_constraints;
        let n = problem.num_lower_vars;
        let big_m = self.config.bigm_config.default_m;

        for i in 0..m {
            let mut cut = MilpConstraint::new(
                &format!("bound_cut_constr_{}", i),
                ConstraintSense::Le,
                big_m,
            );
            cut.add_term(layout.lambda_idx(i), 1.0);
            cut.add_term(layout.primal_slack_idx(i), 1.0);
            milp.add_constraint(cut);
        }

        for j in 0..n {
            let mut cut =
                MilpConstraint::new(&format!("bound_cut_var_{}", j), ConstraintSense::Le, big_m);
            cut.add_term(layout.dual_slack_idx(j), 1.0);
            cut.add_term(layout.y_idx(j), 1.0);
            milp.add_constraint(cut);
        }
    }

    // ─── complementarity count ───────────────────────────────────────

    /// Total number of complementarity pairs =
    ///   (number of lower constraints) + (number of lower variables).
    pub fn compute_complementarity_count(problem: &BilevelProblem) -> usize {
        problem.num_lower_constraints + problem.num_lower_vars
    }

    // ─── constraint qualification ────────────────────────────────────

    /// Perform a lightweight LICQ (Linear Independence Constraint
    /// Qualification) check on the lower-level constraints.
    ///
    /// For an LP lower level, the KKT conditions are always valid when the
    /// LP is feasible (Slater / strong duality hold automatically), so this
    /// is primarily a structural sanity check.
    ///
    /// We examine the lower-level constraint matrix `A` and flag potential
    /// issues such as linearly dependent rows.
    pub fn check_constraint_qualification(&self, problem: &BilevelProblem) -> CqStatus {
        let m = problem.num_lower_constraints;
        let n = problem.num_lower_vars;

        if m == 0 {
            return CqStatus::Satisfied;
        }

        // For LP lower level, KKT is valid whenever the LP has a finite
        // optimum (strong duality).  We only flag issues when:
        //   1. The constraint matrix has more rows than columns (over-
        //      determined), making LICQ structurally impossible.
        //   2. Duplicate or zero rows exist.
        if m > n {
            return CqStatus::Unknown;
        }

        // Check for zero rows.
        let mut row_nnz = vec![0usize; m];
        for entry in &problem.lower_a.entries {
            if entry.row < m && entry.value.abs() > self.config.tolerance {
                row_nnz[entry.row] += 1;
            }
        }
        let has_zero_row = row_nnz.iter().any(|&count| count == 0);
        if has_zero_row {
            return CqStatus::Violated;
        }

        // Check for duplicate rows by comparing non-zero patterns.
        let mut row_patterns: Vec<Vec<(usize, i64)>> = vec![Vec::new(); m];
        for entry in &problem.lower_a.entries {
            if entry.row < m && entry.value.abs() > self.config.tolerance {
                let quantised = (entry.value / self.config.tolerance).round() as i64;
                row_patterns[entry.row].push((entry.col, quantised));
            }
        }
        for pat in &mut row_patterns {
            pat.sort_by_key(|&(c, _)| c);
        }
        for i in 0..m {
            for j in (i + 1)..m {
                if row_patterns[i] == row_patterns[j] {
                    return CqStatus::Violated;
                }
                // Also check if one row is a scalar multiple of the other.
                if !row_patterns[i].is_empty()
                    && row_patterns[i].len() == row_patterns[j].len()
                    && row_patterns[i]
                        .iter()
                        .zip(row_patterns[j].iter())
                        .all(|(&(ci, _), &(cj, _))| ci == cj)
                {
                    let (_, vi0) = row_patterns[i][0];
                    let (_, vj0) = row_patterns[j][0];
                    if vi0 != 0 && vj0 != 0 {
                        let ratio = vi0 as f64 / vj0 as f64;
                        let all_same_ratio = row_patterns[i]
                            .iter()
                            .zip(row_patterns[j].iter())
                            .all(|(&(_, vi), &(_, vj))| {
                                if vj == 0 {
                                    return vi == 0;
                                }
                                ((vi as f64 / vj as f64) - ratio).abs()
                                    < self.config.tolerance * 1e4
                            });
                        if all_same_ratio {
                            return CqStatus::Violated;
                        }
                    }
                }
            }
        }

        CqStatus::Satisfied
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use bicut_types::SparseMatrix;

    /// Helper: build a minimal bilevel problem.
    ///
    /// ```text
    ///   min_{x,y}  -x - y
    ///   s.t.       x + y ≤ 4        (upper constraint)
    ///              y ∈ argmin_{y'} { -y' : y' ≤ 3 + 0·x, y' ≥ 0 }
    /// ```
    ///
    /// So: 1 upper var (x), 1 lower var (y), 1 lower constraint (y ≤ 3).
    fn make_simple_bilevel() -> BilevelProblem {
        let num_upper_vars = 1;
        let num_lower_vars = 1;
        let num_lower_constraints = 1;
        let num_upper_constraints = 1;

        // Lower: A = [1], b = [3], c = [-1] (min -y).
        let mut lower_a = SparseMatrix::new(1, 1);
        lower_a.add_entry(0, 0, 1.0);

        // B = [0] (no coupling in lower RHS).
        let lower_linking_b = SparseMatrix::new(1, 1);

        // Upper constraints: x + y ≤ 4  →  A_upper = [1, 1], b = [4].
        // Columns: 0 → x, 1 → y.
        let mut upper_a = SparseMatrix::new(1, 2);
        upper_a.add_entry(0, 0, 1.0);
        upper_a.add_entry(0, 1, 1.0);

        BilevelProblem {
            upper_obj_c_x: vec![-1.0],
            upper_obj_c_y: vec![-1.0],
            lower_obj_c: vec![-1.0],
            lower_a,
            lower_b: vec![3.0],
            lower_linking_b,
            upper_constraints_a: upper_a,
            upper_constraints_b: vec![4.0],
            num_upper_vars,
            num_lower_vars,
            num_lower_constraints,
            num_upper_constraints,
        }
    }

    /// Helper: bilevel with 2 upper vars, 2 lower vars, 2 lower constraints.
    fn make_two_by_two() -> BilevelProblem {
        let mut lower_a = SparseMatrix::new(2, 2);
        lower_a.add_entry(0, 0, 1.0);
        lower_a.add_entry(0, 1, 0.0);
        lower_a.add_entry(1, 0, 0.0);
        lower_a.add_entry(1, 1, 1.0);

        let mut linking = SparseMatrix::new(2, 2);
        linking.add_entry(0, 0, 1.0);
        linking.add_entry(1, 1, 1.0);

        let upper_a = SparseMatrix::new(0, 4);

        BilevelProblem {
            upper_obj_c_x: vec![1.0, 0.0],
            upper_obj_c_y: vec![0.0, 1.0],
            lower_obj_c: vec![1.0, 1.0],
            lower_a,
            lower_b: vec![5.0, 5.0],
            lower_linking_b: linking,
            upper_constraints_a: upper_a,
            upper_constraints_b: vec![],
            num_upper_vars: 2,
            num_lower_vars: 2,
            num_lower_constraints: 2,
            num_upper_constraints: 0,
        }
    }

    // ─── VarLayout tests ─────────────────────────────────────────────

    #[test]
    fn test_var_layout_simple_bigm() {
        let p = make_simple_bilevel();
        let layout = VarLayout::new(&p, ComplementarityEncoding::BigM);

        assert_eq!(layout.x_start, 0);
        assert_eq!(layout.x_count, 1);
        assert_eq!(layout.y_start, 1);
        assert_eq!(layout.y_count, 1);
        assert_eq!(layout.lambda_start, 2);
        assert_eq!(layout.lambda_count, 1);
        assert_eq!(layout.primal_slack_start, 3);
        assert_eq!(layout.primal_slack_count, 1);
        assert_eq!(layout.dual_slack_start, 4);
        assert_eq!(layout.dual_slack_count, 1);
        assert_eq!(layout.binary_start, 5);
        // 1 constraint comp + 1 variable comp = 2 binary vars
        assert_eq!(layout.binary_count, 2);
        assert_eq!(layout.total_vars, 7);
    }

    #[test]
    fn test_var_layout_sos1_no_binaries() {
        let p = make_simple_bilevel();
        let layout = VarLayout::new(&p, ComplementarityEncoding::SOS1);

        assert_eq!(layout.binary_count, 0);
        assert_eq!(layout.total_vars, 5); // x, y, λ, ps, ds
    }

    #[test]
    fn test_var_layout_indicator_has_binaries() {
        let p = make_simple_bilevel();
        let layout = VarLayout::new(&p, ComplementarityEncoding::Indicator);
        assert_eq!(layout.binary_count, 2);
        assert_eq!(layout.total_vars, 7);
    }

    #[test]
    fn test_var_layout_index_methods() {
        let p = make_two_by_two();
        let layout = VarLayout::new(&p, ComplementarityEncoding::BigM);

        assert_eq!(layout.x_idx(0), 0);
        assert_eq!(layout.x_idx(1), 1);
        assert_eq!(layout.y_idx(0), 2);
        assert_eq!(layout.y_idx(1), 3);
        assert_eq!(layout.lambda_idx(0), 4);
        assert_eq!(layout.lambda_idx(1), 5);
        assert_eq!(layout.primal_slack_idx(0), 6);
        assert_eq!(layout.primal_slack_idx(1), 7);
        assert_eq!(layout.dual_slack_idx(0), 8);
        assert_eq!(layout.dual_slack_idx(1), 9);
        assert_eq!(layout.binary_idx(0), 10);
        // m + n = 2 + 2 = 4 binary vars
        assert_eq!(layout.binary_count, 4);
        assert_eq!(layout.total_vars, 14);
    }

    // ─── complementarity count ───────────────────────────────────────

    #[test]
    fn test_complementarity_count() {
        let p = make_simple_bilevel();
        assert_eq!(KktPass::compute_complementarity_count(&p), 2);

        let p2 = make_two_by_two();
        assert_eq!(KktPass::compute_complementarity_count(&p2), 4);
    }

    // ─── Big-M encoding ──────────────────────────────────────────────

    #[test]
    fn test_bigm_encoding_produces_binary_vars() {
        let p = make_simple_bilevel();
        let config = KktConfig {
            encoding: ComplementarityEncoding::BigM,
            verify_cq: false,
            ..KktConfig::default()
        };
        let pass = KktPass::new(config);
        let result = pass.apply(&p).expect("apply failed");

        assert!(result.num_binary_vars > 0);
        assert_eq!(result.num_binary_vars, 2);
        assert!(result.bigm_set.is_some());

        let num_bin = result
            .milp
            .variables
            .iter()
            .filter(|v| v.var_type == VarType::Binary)
            .count();
        assert_eq!(num_bin, 2);
    }

    // ─── SOS1 encoding ──────────────────────────────────────────────

    #[test]
    fn test_sos1_encoding_produces_sos1_sets() {
        let p = make_simple_bilevel();
        let config = KktConfig {
            encoding: ComplementarityEncoding::SOS1,
            verify_cq: false,
            ..KktConfig::default()
        };
        let pass = KktPass::new(config);
        let result = pass.apply(&p).expect("apply failed");

        // 1 constraint pair + 1 variable pair = 2 SOS1 sets
        assert_eq!(result.milp.sos1_sets.len(), 2);
        assert_eq!(result.num_binary_vars, 0);
        assert!(result.bigm_set.is_none());
    }

    // ─── Indicator encoding ──────────────────────────────────────────

    #[test]
    fn test_indicator_encoding_produces_indicators() {
        let p = make_simple_bilevel();
        let config = KktConfig {
            encoding: ComplementarityEncoding::Indicator,
            verify_cq: false,
            ..KktConfig::default()
        };
        let pass = KktPass::new(config);
        let result = pass.apply(&p).expect("apply failed");

        // 2 per constraint pair + 2 per variable pair = 4 indicator constraints
        assert_eq!(result.milp.indicator_constraints.len(), 4);
        assert_eq!(result.num_binary_vars, 2);
    }

    // ─── CQ checking ────────────────────────────────────────────────

    #[test]
    fn test_cq_satisfied_for_simple_problem() {
        let p = make_simple_bilevel();
        let pass = KktPass::new(KktConfig::default());
        let cq = pass.check_constraint_qualification(&p);
        assert_eq!(cq, CqStatus::Satisfied);
    }

    #[test]
    fn test_cq_violated_for_duplicate_rows() {
        let mut p = make_two_by_two();
        // Make both rows of A identical → linearly dependent.
        p.lower_a = SparseMatrix::new(2, 2);
        p.lower_a.add_entry(0, 0, 1.0);
        p.lower_a.add_entry(0, 1, 2.0);
        p.lower_a.add_entry(1, 0, 1.0);
        p.lower_a.add_entry(1, 1, 2.0);

        let pass = KktPass::new(KktConfig::default());
        let cq = pass.check_constraint_qualification(&p);
        assert_eq!(cq, CqStatus::Violated);
    }

    // ─── full apply round-trip ───────────────────────────────────────

    #[test]
    fn test_full_apply_simple_bigm() {
        let p = make_simple_bilevel();
        let config = KktConfig {
            encoding: ComplementarityEncoding::BigM,
            verify_cq: true,
            add_bound_cuts: false,
            ..KktConfig::default()
        };
        let pass = KktPass::new(config);
        let result = pass.apply(&p).expect("apply failed");

        // Variables: x(1), y(1), λ(1), ps(1), ds(1), z(2) = 7
        assert_eq!(result.milp.num_vars(), 7);
        assert_eq!(result.num_dual_vars, 1);
        assert_eq!(result.num_complementarity_pairs, 2);
        assert_eq!(result.cq_status, CqStatus::Satisfied);

        // Constraints:
        //   upper: 1
        //   primal_feas: 1
        //   stationarity: 1
        //   comp_bigm: 2 (dual) + 2 (variable) = 4
        //   total = 7
        assert_eq!(result.milp.num_constraints(), 7);
    }

    #[test]
    fn test_full_apply_two_by_two_sos1() {
        let p = make_two_by_two();
        let config = KktConfig {
            encoding: ComplementarityEncoding::SOS1,
            verify_cq: true,
            ..KktConfig::default()
        };
        let pass = KktPass::new(config);
        let result = pass.apply(&p).expect("apply failed");

        // Variables: x(2), y(2), λ(2), ps(2), ds(2), binary(0) = 10
        assert_eq!(result.milp.num_vars(), 10);
        assert_eq!(result.num_dual_vars, 2);
        assert_eq!(result.num_complementarity_pairs, 4);
        assert_eq!(result.milp.sos1_sets.len(), 4);

        // Constraints: upper(0) + primal_feas(2) + stationarity(2) = 4
        assert_eq!(result.milp.num_constraints(), 4);
    }

    #[test]
    fn test_validation_rejects_zero_lower_vars() {
        let p = BilevelProblem {
            upper_obj_c_x: vec![1.0],
            upper_obj_c_y: vec![],
            lower_obj_c: vec![],
            lower_a: SparseMatrix::new(0, 0),
            lower_b: vec![],
            lower_linking_b: SparseMatrix::new(0, 1),
            upper_constraints_a: SparseMatrix::new(0, 1),
            upper_constraints_b: vec![],
            num_upper_vars: 1,
            num_lower_vars: 0,
            num_lower_constraints: 0,
            num_upper_constraints: 0,
        };
        let pass = KktPass::new(KktConfig::default());
        assert!(pass.apply(&p).is_err());
    }
}
