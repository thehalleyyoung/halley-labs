//! Strong duality reformulation pass for bilevel LP programs.
//!
//! For a bilevel program whose lower level is a linear program:
//!
//! ```text
//!   y ∈ argmin { c^T y : Ay ≤ b + Bx, y ≥ 0 }
//! ```
//!
//! we replace the lower-level optimality condition with three groups of
//! linear constraints:
//!
//! 1. **Primal feasibility**: `Ay ≤ b + Bx,  y ≥ 0`
//! 2. **Dual feasibility**:   `A^T λ ≥ c,    λ ≥ 0`
//! 3. **Strong duality**:     `c^T y = (b + Bx)^T λ`
//!
//! This avoids complementarity conditions and Big-M parameters entirely,
//! at the cost of introducing bilinear terms `x_i · λ_j` in the strong
//! duality equality whenever the linking matrix `B` is non-zero.  Those
//! terms are handled via McCormick envelope relaxations.

use std::collections::{HashMap, HashSet};

use crate::*;

// ═══════════════════════════════════════════════════════════════════════════
// Pipeline / KKT types (defined locally until those modules are created)
// ═══════════════════════════════════════════════════════════════════════════

/// Variable type in the single-level MILP formulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VarType {
    Continuous,
    Integer,
    Binary,
}

/// SOS-1 constraint: at most one member variable may be non-zero.
#[derive(Debug, Clone)]
pub struct Sos1Set {
    pub members: Vec<usize>,
    pub weights: Vec<f64>,
    pub name: String,
}

/// Indicator constraint: if `binary_var == active_value` then the linear
/// inequality described by `coeffs`, `sense`, `rhs` is enforced.
#[derive(Debug, Clone)]
pub struct IndicatorConstraint {
    pub binary_var: usize,
    pub active_value: bool,
    pub coeffs: Vec<(usize, f64)>,
    pub sense: ConstraintSense,
    pub rhs: f64,
    pub name: String,
}

/// Mixed-Integer Linear Program produced by the reformulation.
#[derive(Debug, Clone)]
pub struct MilpProblem {
    pub direction: OptDirection,
    pub obj_coeffs: Vec<f64>,
    pub constraint_matrix: SparseMatrix,
    pub constraint_rhs: Vec<f64>,
    pub constraint_senses: Vec<ConstraintSense>,
    pub var_types: Vec<VarType>,
    pub var_lower: Vec<f64>,
    pub var_upper: Vec<f64>,
    pub var_names: Vec<String>,
    pub constraint_names: Vec<String>,
    pub sos1_sets: Vec<Sos1Set>,
    pub indicators: Vec<IndicatorConstraint>,
    pub num_vars: usize,
    pub num_constraints: usize,
}

impl MilpProblem {
    fn empty(num_vars: usize, direction: OptDirection) -> Self {
        Self {
            direction,
            obj_coeffs: vec![0.0; num_vars],
            constraint_matrix: SparseMatrix {
                rows: 0,
                cols: num_vars,
                entries: Vec::new(),
            },
            constraint_rhs: Vec::new(),
            constraint_senses: Vec::new(),
            var_types: vec![VarType::Continuous; num_vars],
            var_lower: vec![0.0; num_vars],
            var_upper: vec![f64::INFINITY; num_vars],
            var_names: (0..num_vars).map(|i| format!("v{}", i)).collect(),
            constraint_names: Vec::new(),
            sos1_sets: Vec::new(),
            indicators: Vec::new(),
            num_vars,
            num_constraints: 0,
        }
    }
}

/// Variable layout from the KKT pass (basic x/y index tracking).
#[derive(Debug, Clone)]
pub struct VarLayout {
    pub x_start: usize,
    pub x_count: usize,
    pub y_start: usize,
    pub y_count: usize,
    pub total_vars: usize,
}

/// Compiler configuration stub.
#[derive(Debug, Clone)]
pub struct CompilerConfig {
    pub reformulation: ReformulationType,
    pub big_m: f64,
    pub presolve: bool,
    pub tolerance: f64,
}

impl Default for CompilerConfig {
    fn default() -> Self {
        Self {
            reformulation: ReformulationType::StrongDuality,
            big_m: 1e6,
            presolve: true,
            tolerance: DEFAULT_TOLERANCE,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Strong Duality Configuration
// ═══════════════════════════════════════════════════════════════════════════

/// Tunables for the strong-duality reformulation pass.
#[derive(Debug, Clone)]
pub struct StrongDualityConfig {
    /// Numerical zero tolerance.
    pub tolerance: f64,
    /// Whether to introduce McCormick auxiliary variables for bilinear
    /// `x_i · λ_j` terms in the strong duality equality.
    pub add_mccormick_for_bilinear: bool,
    /// Number of partitions used when piecewise-McCormick is enabled.
    pub mccormick_partitions: usize,
    /// If `true`, verify that the lower level is a pure LP before
    /// applying the reformulation (reject integer lower levels).
    pub verify_lp_lower_level: bool,
}

impl Default for StrongDualityConfig {
    fn default() -> Self {
        Self {
            tolerance: DEFAULT_TOLERANCE,
            add_mccormick_for_bilinear: true,
            mccormick_partitions: 4,
            verify_lp_lower_level: true,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Strong Duality Result
// ═══════════════════════════════════════════════════════════════════════════

/// Output of the strong-duality reformulation pass.
#[derive(Debug, Clone)]
pub struct StrongDualityResult {
    /// The single-level MILP (or LP) equivalent of the bilevel problem.
    pub milp: MilpProblem,
    /// Number of dual variables (λ) introduced.
    pub num_dual_vars: usize,
    /// Number of bilinear `x_i · λ_j` terms in the strong duality
    /// equality before linearisation.
    pub num_bilinear_terms: usize,
    /// Whether McCormick envelopes were actually added.
    pub mccormick_added: bool,
    /// Non-fatal warnings produced during the reformulation.
    pub warnings: Vec<String>,
}

// ═══════════════════════════════════════════════════════════════════════════
// Dual Variable Layout
// ═══════════════════════════════════════════════════════════════════════════

/// Maps each variable group to a contiguous index range inside the
/// single-level MILP.
///
/// Layout: `[ x_0 .. x_{n-1} | y_0 .. y_{m-1} | λ_0 .. λ_{p-1} | w_... ]`
#[derive(Debug, Clone)]
pub struct DualVarLayout {
    pub x_start: usize,
    pub x_count: usize,
    pub y_start: usize,
    pub y_count: usize,
    pub lambda_start: usize,
    pub lambda_count: usize,
    pub mccormick_aux_start: usize,
    pub mccormick_aux_count: usize,
    pub total_vars: usize,
}

impl DualVarLayout {
    /// Build the initial layout (without McCormick auxiliaries).
    pub fn new(problem: &BilevelProblem) -> Self {
        let x_start = 0;
        let x_count = problem.num_upper_vars;
        let y_start = x_start + x_count;
        let y_count = problem.num_lower_vars;
        let lambda_start = y_start + y_count;
        let lambda_count = problem.num_lower_constraints;
        let total = lambda_start + lambda_count;
        Self {
            x_start,
            x_count,
            y_start,
            y_count,
            lambda_start,
            lambda_count,
            mccormick_aux_start: total,
            mccormick_aux_count: 0,
            total_vars: total,
        }
    }

    /// Return the MILP column index of the `i`-th upper-level variable.
    #[inline]
    pub fn x_idx(&self, i: usize) -> usize {
        debug_assert!(
            i < self.x_count,
            "x index {} out of range {}",
            i,
            self.x_count
        );
        self.x_start + i
    }

    /// Return the MILP column index of the `i`-th lower-level variable.
    #[inline]
    pub fn y_idx(&self, i: usize) -> usize {
        debug_assert!(
            i < self.y_count,
            "y index {} out of range {}",
            i,
            self.y_count
        );
        self.y_start + i
    }

    /// Return the MILP column index of the `i`-th dual variable.
    #[inline]
    pub fn lambda_idx(&self, i: usize) -> usize {
        debug_assert!(
            i < self.lambda_count,
            "lambda index {} out of range {}",
            i,
            self.lambda_count
        );
        self.lambda_start + i
    }

    /// Extend the layout with `count` McCormick auxiliary variables.
    pub fn with_mccormick(mut self, count: usize) -> Self {
        self.mccormick_aux_count = count;
        self.total_vars = self.mccormick_aux_start + count;
        self
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Bilinear Term
// ═══════════════════════════════════════════════════════════════════════════

/// A bilinear product `coefficient · x_{x_var} · λ_{lambda_var}` arising
/// from the expansion of `x^T B^T λ` in the strong duality equality.
#[derive(Debug, Clone)]
pub struct BilinearTerm {
    /// Index into the *local* x-variable space (0-based).
    pub x_var: usize,
    /// Index into the *local* λ-variable space (0-based).
    pub lambda_var: usize,
    /// Scalar coefficient (from B).
    pub coefficient: f64,
    /// Lower bound on `x_{x_var}`.
    pub x_lower: f64,
    /// Upper bound on `x_{x_var}`.
    pub x_upper: f64,
    /// Lower bound on `λ_{lambda_var}`.
    pub lambda_lower: f64,
    /// Upper bound on `λ_{lambda_var}`.
    pub lambda_upper: f64,
}

// ═══════════════════════════════════════════════════════════════════════════
// McCormick Envelope
// ═══════════════════════════════════════════════════════════════════════════

/// Internal representation of a single linear constraint produced by
/// McCormick linearisation.
#[derive(Debug, Clone)]
struct LinearConstraintRepr {
    coeffs: Vec<(usize, f64)>,
    sense: ConstraintSense,
    rhs: f64,
}

/// McCormick relaxation envelope for a single bilinear term `w = x · y`.
///
/// Introduces auxiliary variable `w` (at `aux_var_idx`) and four linear
/// constraints that form a convex relaxation of the product.
#[derive(Debug, Clone)]
pub struct McCormickEnvelope {
    /// MILP column index of the auxiliary variable `w`.
    pub aux_var_idx: usize,
    /// MILP column index of the first factor.
    pub x_var: usize,
    /// MILP column index of the second factor.
    pub y_var: usize,
    /// `(lower, upper)` bounds on `x`.
    pub x_bounds: (f64, f64),
    /// `(lower, upper)` bounds on `y`.
    pub y_bounds: (f64, f64),
}

impl McCormickEnvelope {
    /// Generate the four McCormick envelope constraints for `w = x · y`:
    ///
    /// ```text
    ///   w ≥ x_L · y + x · y_L − x_L · y_L
    ///   w ≥ x_U · y + x · y_U − x_U · y_U
    ///   w ≤ x_U · y + x · y_L − x_U · y_L
    ///   w ≤ x_L · y + x · y_U − x_L · y_U
    /// ```
    pub fn generate_constraints(&self) -> Vec<LinearConstraintRepr> {
        let w = self.aux_var_idx;
        let x = self.x_var;
        let y = self.y_var;
        let (x_l, x_u) = self.x_bounds;
        let (y_l, y_u) = self.y_bounds;

        vec![
            // w ≥ x_L·y + x·y_L − x_L·y_L
            // ⟹ w − x_L·y − y_L·x ≥ −x_L·y_L
            LinearConstraintRepr {
                coeffs: vec![(w, 1.0), (y, -x_l), (x, -y_l)],
                sense: ConstraintSense::Ge,
                rhs: -x_l * y_l,
            },
            // w ≥ x_U·y + x·y_U − x_U·y_U
            LinearConstraintRepr {
                coeffs: vec![(w, 1.0), (y, -x_u), (x, -y_u)],
                sense: ConstraintSense::Ge,
                rhs: -x_u * y_u,
            },
            // w ≤ x_U·y + x·y_L − x_U·y_L
            LinearConstraintRepr {
                coeffs: vec![(w, 1.0), (y, -x_u), (x, -y_l)],
                sense: ConstraintSense::Le,
                rhs: -x_u * y_l,
            },
            // w ≤ x_L·y + x·y_U − x_L·y_U
            LinearConstraintRepr {
                coeffs: vec![(w, 1.0), (y, -x_l), (x, -y_u)],
                sense: ConstraintSense::Le,
                rhs: -x_l * y_u,
            },
        ]
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Strong Duality Pass
// ═══════════════════════════════════════════════════════════════════════════

/// The pass itself: holds configuration and implements the reformulation.
#[derive(Debug, Clone)]
pub struct StrongDualityPass {
    config: StrongDualityConfig,
}

impl StrongDualityPass {
    /// Create a new pass with the given configuration.
    pub fn new(config: StrongDualityConfig) -> Self {
        Self { config }
    }

    // ──────────────────────────────────────────────────────────────────────
    // Public entry point
    // ──────────────────────────────────────────────────────────────────────

    /// Apply the strong-duality reformulation to `problem`, producing a
    /// single-level MILP.
    pub fn apply(&self, problem: &BilevelProblem) -> Result<StrongDualityResult, CompilerError> {
        // 1. Validate that the lower level is indeed an LP.
        if self.config.verify_lp_lower_level && !Self::verify_lp_lower_level(problem) {
            return Err(CompilerError::InvalidProblem(
                "Strong duality reformulation requires an LP lower level \
                 (dimensions inconsistent or lower level empty)"
                    .into(),
            ));
        }

        // 2. Identify bilinear terms and finalise the variable layout.
        let bilinear_terms = Self::identify_bilinear_terms(problem);
        let num_mccormick = if self.config.add_mccormick_for_bilinear {
            bilinear_terms.len()
        } else {
            0
        };
        let layout = DualVarLayout::new(problem).with_mccormick(num_mccormick);

        // 3. Initialise the MILP shell.
        let mut milp = MilpProblem::empty(layout.total_vars, OptDirection::Minimize);
        let mut entries: Vec<SparseEntry> = Vec::new();
        let mut rhs: Vec<f64> = Vec::new();
        let mut senses: Vec<ConstraintSense> = Vec::new();
        let mut names: Vec<String> = Vec::new();
        let mut row: usize = 0;
        let mut warnings: Vec<String> = Vec::new();

        // Name and bound the x, y, λ variables.
        for i in 0..layout.x_count {
            milp.var_names[layout.x_idx(i)] = format!("x_{}", i);
            milp.var_lower[layout.x_idx(i)] = 0.0;
            milp.var_upper[layout.x_idx(i)] = f64::INFINITY;
        }
        for j in 0..layout.y_count {
            milp.var_names[layout.y_idx(j)] = format!("y_{}", j);
            milp.var_lower[layout.y_idx(j)] = 0.0;
            milp.var_upper[layout.y_idx(j)] = f64::INFINITY;
        }
        for k in 0..layout.lambda_count {
            milp.var_names[layout.lambda_idx(k)] = format!("lambda_{}", k);
            milp.var_lower[layout.lambda_idx(k)] = 0.0;
            milp.var_upper[layout.lambda_idx(k)] = f64::INFINITY;
        }

        // 4–6. Build constraint groups.
        self.add_primal_feasibility(
            problem,
            &layout,
            &mut entries,
            &mut rhs,
            &mut senses,
            &mut names,
            &mut row,
        );
        self.add_dual_feasibility(
            problem,
            &layout,
            &mut entries,
            &mut rhs,
            &mut senses,
            &mut names,
            &mut row,
        );
        self.add_strong_duality_equality(
            problem,
            &layout,
            &bilinear_terms,
            &mut entries,
            &mut rhs,
            &mut senses,
            &mut names,
            &mut row,
            &mut warnings,
        );

        // 7. McCormick envelopes.
        let mccormick_added =
            if self.config.add_mccormick_for_bilinear && !bilinear_terms.is_empty() {
                self.add_mccormick_envelopes(
                    &bilinear_terms,
                    &layout,
                    &mut milp,
                    &mut entries,
                    &mut rhs,
                    &mut senses,
                    &mut names,
                    &mut row,
                );
                true
            } else {
                if !bilinear_terms.is_empty() {
                    warnings.push(format!(
                        "Strong duality equality has {} bilinear x·λ terms but \
                         McCormick envelopes are disabled — MILP is not a valid relaxation",
                        bilinear_terms.len()
                    ));
                }
                false
            };

        // 8–9. Upper-level objective and constraints.
        self.build_upper_objective(problem, &layout, &mut milp);
        self.add_upper_constraints(
            problem,
            &layout,
            &mut entries,
            &mut rhs,
            &mut senses,
            &mut names,
            &mut row,
        );

        // Assemble the constraint matrix.
        milp.constraint_matrix = SparseMatrix {
            rows: row,
            cols: layout.total_vars,
            entries,
        };
        milp.constraint_rhs = rhs;
        milp.constraint_senses = senses;
        milp.constraint_names = names;
        milp.num_constraints = row;

        Ok(StrongDualityResult {
            milp,
            num_dual_vars: layout.lambda_count,
            num_bilinear_terms: bilinear_terms.len(),
            mccormick_added,
            warnings,
        })
    }

    // ──────────────────────────────────────────────────────────────────────
    // Validation
    // ──────────────────────────────────────────────────────────────────────

    /// Return `true` when the lower level of `problem` is a well-formed LP.
    ///
    /// The [`BilevelProblem`] struct inherently represents an LP lower
    /// level (continuous variables, linear constraints).  This check
    /// verifies that all dimension fields are mutually consistent and
    /// non-trivial.
    pub fn verify_lp_lower_level(problem: &BilevelProblem) -> bool {
        if problem.num_lower_vars == 0 || problem.num_lower_constraints == 0 {
            return false;
        }
        if problem.lower_obj_c.len() != problem.num_lower_vars {
            return false;
        }
        if problem.lower_a.rows != problem.num_lower_constraints {
            return false;
        }
        if problem.lower_a.cols != problem.num_lower_vars {
            return false;
        }
        if problem.lower_b.len() != problem.num_lower_constraints {
            return false;
        }
        if problem.lower_linking_b.rows != problem.num_lower_constraints {
            return false;
        }
        if problem.lower_linking_b.cols != problem.num_upper_vars {
            return false;
        }
        true
    }

    // ──────────────────────────────────────────────────────────────────────
    // Primal feasibility: Ay ≤ b + Bx  ⟹  Ay − Bx ≤ b,  y ≥ 0
    // ──────────────────────────────────────────────────────────────────────

    fn add_primal_feasibility(
        &self,
        problem: &BilevelProblem,
        layout: &DualVarLayout,
        entries: &mut Vec<SparseEntry>,
        rhs: &mut Vec<f64>,
        senses: &mut Vec<ConstraintSense>,
        names: &mut Vec<String>,
        row: &mut usize,
    ) {
        for i in 0..problem.num_lower_constraints {
            // +A[i,j] · y_j
            for e in &problem.lower_a.entries {
                if e.row == i && e.value.abs() > self.config.tolerance {
                    entries.push(SparseEntry {
                        row: *row,
                        col: layout.y_idx(e.col),
                        value: e.value,
                    });
                }
            }
            // −B[i,k] · x_k   (move Bx to the left)
            for e in &problem.lower_linking_b.entries {
                if e.row == i && e.value.abs() > self.config.tolerance {
                    entries.push(SparseEntry {
                        row: *row,
                        col: layout.x_idx(e.col),
                        value: -e.value,
                    });
                }
            }
            rhs.push(problem.lower_b[i]);
            senses.push(ConstraintSense::Le);
            names.push(format!("primal_feas_{}", i));
            *row += 1;
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // Dual feasibility: A^T λ ≥ c,  λ ≥ 0
    // ──────────────────────────────────────────────────────────────────────

    fn add_dual_feasibility(
        &self,
        problem: &BilevelProblem,
        layout: &DualVarLayout,
        entries: &mut Vec<SparseEntry>,
        rhs: &mut Vec<f64>,
        senses: &mut Vec<ConstraintSense>,
        names: &mut Vec<String>,
        row: &mut usize,
    ) {
        // For each lower variable j:  Σ_i A[i,j] · λ_i  ≥  c[j]
        for j in 0..problem.num_lower_vars {
            for e in &problem.lower_a.entries {
                if e.col == j && e.value.abs() > self.config.tolerance {
                    entries.push(SparseEntry {
                        row: *row,
                        col: layout.lambda_idx(e.row),
                        value: e.value,
                    });
                }
            }
            rhs.push(problem.lower_obj_c[j]);
            senses.push(ConstraintSense::Ge);
            names.push(format!("dual_feas_{}", j));
            *row += 1;
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // Strong duality: c^T y = b^T λ + x^T B^T λ
    //   ⟹  c^T y − b^T λ − Σ coeff·w_{ij}  = 0
    // (where w_{ij} ≈ x_i · λ_j via McCormick, or terms are dropped)
    // ──────────────────────────────────────────────────────────────────────

    fn add_strong_duality_equality(
        &self,
        problem: &BilevelProblem,
        layout: &DualVarLayout,
        bilinear_terms: &[BilinearTerm],
        entries: &mut Vec<SparseEntry>,
        rhs: &mut Vec<f64>,
        senses: &mut Vec<ConstraintSense>,
        names: &mut Vec<String>,
        row: &mut usize,
        warnings: &mut Vec<String>,
    ) {
        // +c[j] · y_j
        for j in 0..problem.num_lower_vars {
            let cj = problem.lower_obj_c[j];
            if cj.abs() > self.config.tolerance {
                entries.push(SparseEntry {
                    row: *row,
                    col: layout.y_idx(j),
                    value: cj,
                });
            }
        }

        // −b[i] · λ_i
        for i in 0..problem.num_lower_constraints {
            let bi = problem.lower_b[i];
            if bi.abs() > self.config.tolerance {
                entries.push(SparseEntry {
                    row: *row,
                    col: layout.lambda_idx(i),
                    value: -bi,
                });
            }
        }

        // Bilinear block: −Σ coefficient · w_{idx}
        if self.config.add_mccormick_for_bilinear {
            for (idx, term) in bilinear_terms.iter().enumerate() {
                let w_col = layout.mccormick_aux_start + idx;
                entries.push(SparseEntry {
                    row: *row,
                    col: w_col,
                    value: -term.coefficient,
                });
            }
        } else if !bilinear_terms.is_empty() {
            warnings.push(
                "Bilinear x·λ terms in the strong duality equality were \
                 dropped (McCormick disabled)"
                    .into(),
            );
        }

        rhs.push(0.0);
        senses.push(ConstraintSense::Eq);
        names.push("strong_duality".into());
        *row += 1;
    }

    // ──────────────────────────────────────────────────────────────────────
    // McCormick envelopes for each bilinear term
    // ──────────────────────────────────────────────────────────────────────

    fn add_mccormick_envelopes(
        &self,
        bilinear_terms: &[BilinearTerm],
        layout: &DualVarLayout,
        milp: &mut MilpProblem,
        entries: &mut Vec<SparseEntry>,
        rhs: &mut Vec<f64>,
        senses: &mut Vec<ConstraintSense>,
        names: &mut Vec<String>,
        row: &mut usize,
    ) {
        for (idx, term) in bilinear_terms.iter().enumerate() {
            let w_col = layout.mccormick_aux_start + idx;

            // Name and bound the auxiliary variable.
            milp.var_names[w_col] = format!("w_x{}_lam{}", term.x_var, term.lambda_var);
            let products = [
                term.x_lower * term.lambda_lower,
                term.x_lower * term.lambda_upper,
                term.x_upper * term.lambda_lower,
                term.x_upper * term.lambda_upper,
            ];
            milp.var_lower[w_col] = products.iter().copied().fold(f64::INFINITY, f64::min);
            milp.var_upper[w_col] = products.iter().copied().fold(f64::NEG_INFINITY, f64::max);

            let envelope = McCormickEnvelope {
                aux_var_idx: w_col,
                x_var: layout.x_idx(term.x_var),
                y_var: layout.lambda_idx(term.lambda_var),
                x_bounds: (term.x_lower, term.x_upper),
                y_bounds: (term.lambda_lower, term.lambda_upper),
            };

            for (ci, mc) in envelope.generate_constraints().iter().enumerate() {
                for &(col, val) in &mc.coeffs {
                    if val.abs() > self.config.tolerance {
                        entries.push(SparseEntry {
                            row: *row,
                            col,
                            value: val,
                        });
                    }
                }
                rhs.push(mc.rhs);
                senses.push(mc.sense);
                names.push(format!("mc_x{}_lam{}_{}", term.x_var, term.lambda_var, ci));
                *row += 1;
            }
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // Upper-level objective: min  c_x^T x + c_y^T y
    // ──────────────────────────────────────────────────────────────────────

    fn build_upper_objective(
        &self,
        problem: &BilevelProblem,
        layout: &DualVarLayout,
        milp: &mut MilpProblem,
    ) {
        for i in 0..problem.num_upper_vars.min(problem.upper_obj_c_x.len()) {
            milp.obj_coeffs[layout.x_idx(i)] = problem.upper_obj_c_x[i];
        }
        for j in 0..problem.num_lower_vars.min(problem.upper_obj_c_y.len()) {
            milp.obj_coeffs[layout.y_idx(j)] = problem.upper_obj_c_y[j];
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // Upper-level constraints (joint in x and y)
    // ──────────────────────────────────────────────────────────────────────

    fn add_upper_constraints(
        &self,
        problem: &BilevelProblem,
        layout: &DualVarLayout,
        entries: &mut Vec<SparseEntry>,
        rhs: &mut Vec<f64>,
        senses: &mut Vec<ConstraintSense>,
        names: &mut Vec<String>,
        row: &mut usize,
    ) {
        let n_x = problem.num_upper_vars;
        for i in 0..problem.num_upper_constraints {
            for e in &problem.upper_constraints_a.entries {
                if e.row == i && e.value.abs() > self.config.tolerance {
                    let col = if e.col < n_x {
                        layout.x_idx(e.col)
                    } else {
                        layout.y_idx(e.col - n_x)
                    };
                    entries.push(SparseEntry {
                        row: *row,
                        col,
                        value: e.value,
                    });
                }
            }
            rhs.push(problem.upper_constraints_b[i]);
            senses.push(ConstraintSense::Le);
            names.push(format!("upper_{}", i));
            *row += 1;
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // Bilinear-term analysis
    // ──────────────────────────────────────────────────────────────────────

    /// Identify all bilinear `x_i · λ_j` terms from the expansion of
    /// `x^T B^T λ`.  Each non-zero entry `B[j, i]` contributes a term
    /// `B[j,i] · x_i · λ_j`.  Duplicate `(i, j)` pairs arising from
    /// multiple B-entries are merged by summing coefficients.
    pub fn identify_bilinear_terms(problem: &BilevelProblem) -> Vec<BilinearTerm> {
        let mut coeff_map: HashMap<(usize, usize), f64> = HashMap::new();
        for e in &problem.lower_linking_b.entries {
            if e.value.abs() > DEFAULT_TOLERANCE {
                // B[row=j, col=i]  →  term  B[j,i] · x_i · λ_j
                *coeff_map.entry((e.col, e.row)).or_insert(0.0) += e.value;
            }
        }

        let mut terms: Vec<BilinearTerm> = coeff_map
            .into_iter()
            .filter(|(_, c)| c.abs() > DEFAULT_TOLERANCE)
            .map(|((x_local, lam_local), coeff)| BilinearTerm {
                x_var: x_local,
                lambda_var: lam_local,
                coefficient: coeff,
                x_lower: 0.0,
                x_upper: 1e6,
                lambda_lower: 0.0,
                lambda_upper: 1e6,
            })
            .collect();

        // Deterministic ordering for reproducibility.
        terms.sort_by(|a, b| {
            a.x_var
                .cmp(&b.x_var)
                .then_with(|| a.lambda_var.cmp(&b.lambda_var))
        });
        terms
    }

    /// Count bilinear terms without allocating the full term list.
    pub fn count_bilinear_terms(problem: &BilevelProblem) -> usize {
        let mut pairs: HashSet<(usize, usize)> = HashSet::new();
        for e in &problem.lower_linking_b.entries {
            if e.value.abs() > DEFAULT_TOLERANCE {
                pairs.insert((e.col, e.row));
            }
        }
        pairs.len()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────

    /// Small bilevel LP:
    ///   min_{x,y}  x + y
    ///   s.t.       x + y ≤ 10          (upper)
    ///              y ∈ argmin { −y : y ≤ 5 + x, y ≥ 0 }
    ///
    /// Lower: A = [[1]], b = [5], B = [[1]], c = [−1].
    fn small_bilevel() -> BilevelProblem {
        BilevelProblem {
            upper_obj_c_x: vec![1.0],
            upper_obj_c_y: vec![1.0],
            lower_obj_c: vec![-1.0],
            lower_a: SparseMatrix {
                rows: 1,
                cols: 1,
                entries: vec![SparseEntry {
                    row: 0,
                    col: 0,
                    value: 1.0,
                }],
            },
            lower_b: vec![5.0],
            lower_linking_b: SparseMatrix {
                rows: 1,
                cols: 1,
                entries: vec![SparseEntry {
                    row: 0,
                    col: 0,
                    value: 1.0,
                }],
            },
            upper_constraints_a: SparseMatrix {
                rows: 1,
                cols: 2,
                entries: vec![
                    SparseEntry {
                        row: 0,
                        col: 0,
                        value: 1.0,
                    },
                    SparseEntry {
                        row: 0,
                        col: 1,
                        value: 1.0,
                    },
                ],
            },
            upper_constraints_b: vec![10.0],
            num_upper_vars: 1,
            num_lower_vars: 1,
            num_lower_constraints: 1,
            num_upper_constraints: 1,
        }
    }

    /// Bilevel LP whose linking matrix B is zero (no bilinear terms).
    fn bilevel_no_linking() -> BilevelProblem {
        BilevelProblem {
            upper_obj_c_x: vec![1.0],
            upper_obj_c_y: vec![2.0],
            lower_obj_c: vec![3.0],
            lower_a: SparseMatrix {
                rows: 1,
                cols: 1,
                entries: vec![SparseEntry {
                    row: 0,
                    col: 0,
                    value: 1.0,
                }],
            },
            lower_b: vec![10.0],
            lower_linking_b: SparseMatrix {
                rows: 1,
                cols: 1,
                entries: vec![],
            },
            upper_constraints_a: SparseMatrix {
                rows: 0,
                cols: 2,
                entries: vec![],
            },
            upper_constraints_b: vec![],
            num_upper_vars: 1,
            num_lower_vars: 1,
            num_lower_constraints: 1,
            num_upper_constraints: 0,
        }
    }

    /// Bilevel LP with 2 upper vars, 2 lower vars, 2 lower constraints,
    /// and a dense linking matrix to stress-test bilinear identification.
    fn bilevel_2x2() -> BilevelProblem {
        BilevelProblem {
            upper_obj_c_x: vec![1.0, 0.5],
            upper_obj_c_y: vec![2.0, 1.0],
            lower_obj_c: vec![1.0, -1.0],
            lower_a: SparseMatrix {
                rows: 2,
                cols: 2,
                entries: vec![
                    SparseEntry {
                        row: 0,
                        col: 0,
                        value: 2.0,
                    },
                    SparseEntry {
                        row: 0,
                        col: 1,
                        value: 1.0,
                    },
                    SparseEntry {
                        row: 1,
                        col: 0,
                        value: 1.0,
                    },
                    SparseEntry {
                        row: 1,
                        col: 1,
                        value: 3.0,
                    },
                ],
            },
            lower_b: vec![8.0, 12.0],
            lower_linking_b: SparseMatrix {
                rows: 2,
                cols: 2,
                entries: vec![
                    SparseEntry {
                        row: 0,
                        col: 0,
                        value: 1.0,
                    },
                    SparseEntry {
                        row: 0,
                        col: 1,
                        value: 0.5,
                    },
                    SparseEntry {
                        row: 1,
                        col: 0,
                        value: -1.0,
                    },
                ],
            },
            upper_constraints_a: SparseMatrix {
                rows: 1,
                cols: 4,
                entries: vec![
                    SparseEntry {
                        row: 0,
                        col: 0,
                        value: 1.0,
                    },
                    SparseEntry {
                        row: 0,
                        col: 1,
                        value: 1.0,
                    },
                    SparseEntry {
                        row: 0,
                        col: 2,
                        value: 1.0,
                    },
                    SparseEntry {
                        row: 0,
                        col: 3,
                        value: 1.0,
                    },
                ],
            },
            upper_constraints_b: vec![20.0],
            num_upper_vars: 2,
            num_lower_vars: 2,
            num_lower_constraints: 2,
            num_upper_constraints: 1,
        }
    }

    // ── test cases ──────────────────────────────────────────────────────

    #[test]
    fn test_verify_lp_lower_level() {
        let good = small_bilevel();
        assert!(StrongDualityPass::verify_lp_lower_level(&good));

        // Mismatched objective length.
        let mut bad_c = small_bilevel();
        bad_c.lower_obj_c = vec![];
        assert!(!StrongDualityPass::verify_lp_lower_level(&bad_c));

        // Mismatched A-matrix column count.
        let mut bad_a = small_bilevel();
        bad_a.lower_a.cols = 99;
        assert!(!StrongDualityPass::verify_lp_lower_level(&bad_a));

        // Zero lower vars.
        let mut bad_zero = small_bilevel();
        bad_zero.num_lower_vars = 0;
        assert!(!StrongDualityPass::verify_lp_lower_level(&bad_zero));
    }

    #[test]
    fn test_dual_var_layout_construction() {
        let prob = small_bilevel();
        let layout = DualVarLayout::new(&prob);

        assert_eq!(layout.x_start, 0);
        assert_eq!(layout.x_count, 1);
        assert_eq!(layout.y_start, 1);
        assert_eq!(layout.y_count, 1);
        assert_eq!(layout.lambda_start, 2);
        assert_eq!(layout.lambda_count, 1);
        assert_eq!(layout.total_vars, 3);
        assert_eq!(layout.mccormick_aux_count, 0);

        assert_eq!(layout.x_idx(0), 0);
        assert_eq!(layout.y_idx(0), 1);
        assert_eq!(layout.lambda_idx(0), 2);

        let ext = layout.with_mccormick(5);
        assert_eq!(ext.mccormick_aux_start, 3);
        assert_eq!(ext.mccormick_aux_count, 5);
        assert_eq!(ext.total_vars, 8);

        // 2×2 problem.
        let prob2 = bilevel_2x2();
        let lay2 = DualVarLayout::new(&prob2);
        assert_eq!(lay2.x_count, 2);
        assert_eq!(lay2.y_count, 2);
        assert_eq!(lay2.lambda_count, 2);
        assert_eq!(lay2.total_vars, 6);
        assert_eq!(lay2.x_idx(1), 1);
        assert_eq!(lay2.y_idx(1), 3);
        assert_eq!(lay2.lambda_idx(1), 5);
    }

    #[test]
    fn test_identify_bilinear_terms() {
        // small_bilevel has B = [[1]], so one term: 1.0 · x_0 · λ_0.
        let terms = StrongDualityPass::identify_bilinear_terms(&small_bilevel());
        assert_eq!(terms.len(), 1);
        assert_eq!(terms[0].x_var, 0);
        assert_eq!(terms[0].lambda_var, 0);
        assert!((terms[0].coefficient - 1.0).abs() < 1e-12);

        // No linking → no bilinear terms.
        let terms_none = StrongDualityPass::identify_bilinear_terms(&bilevel_no_linking());
        assert!(terms_none.is_empty());
        assert_eq!(
            StrongDualityPass::count_bilinear_terms(&bilevel_no_linking()),
            0
        );

        // 2×2 linking matrix has 3 non-zero entries giving 3 terms:
        //   B[0,0]=1   → x_0·λ_0
        //   B[0,1]=0.5 → x_1·λ_0
        //   B[1,0]=−1  → x_0·λ_1
        let terms2 = StrongDualityPass::identify_bilinear_terms(&bilevel_2x2());
        assert_eq!(terms2.len(), 3);
        assert_eq!(StrongDualityPass::count_bilinear_terms(&bilevel_2x2()), 3);
        // Sorted by (x_var, lambda_var).
        assert_eq!((terms2[0].x_var, terms2[0].lambda_var), (0, 0));
        assert_eq!((terms2[1].x_var, terms2[1].lambda_var), (0, 1));
        assert_eq!((terms2[2].x_var, terms2[2].lambda_var), (1, 0));
    }

    #[test]
    fn test_mccormick_envelope_constraints() {
        let env = McCormickEnvelope {
            aux_var_idx: 10,
            x_var: 0,
            y_var: 1,
            x_bounds: (2.0, 6.0),
            y_bounds: (1.0, 4.0),
        };
        let cs = env.generate_constraints();
        assert_eq!(cs.len(), 4);

        // Constraint 0:  w − x_L·y − y_L·x ≥ −x_L·y_L
        //   w − 2·y − 1·x ≥ −2
        assert_eq!(cs[0].sense, ConstraintSense::Ge);
        assert!((cs[0].rhs - (-2.0)).abs() < 1e-12);
        assert_eq!(cs[0].coeffs.len(), 3);
        // (w, 1)
        assert_eq!(cs[0].coeffs[0], (10, 1.0));
        // (y, -x_L = -2)
        assert!((cs[0].coeffs[1].1 - (-2.0)).abs() < 1e-12);
        // (x, -y_L = -1)
        assert!((cs[0].coeffs[2].1 - (-1.0)).abs() < 1e-12);

        // Constraint 1:  w − x_U·y − y_U·x ≥ −x_U·y_U = −24
        assert_eq!(cs[1].sense, ConstraintSense::Ge);
        assert!((cs[1].rhs - (-24.0)).abs() < 1e-12);

        // Constraint 2:  w − x_U·y − y_L·x ≤ −x_U·y_L = −6
        assert_eq!(cs[2].sense, ConstraintSense::Le);
        assert!((cs[2].rhs - (-6.0)).abs() < 1e-12);

        // Constraint 3:  w − x_L·y − y_U·x ≤ −x_L·y_U = −8
        assert_eq!(cs[3].sense, ConstraintSense::Le);
        assert!((cs[3].rhs - (-8.0)).abs() < 1e-12);
    }

    #[test]
    fn test_strong_duality_no_bilinear() {
        // B = 0  ⟹  strong duality is the linear equality  c^T y − b^T λ = 0.
        let prob = bilevel_no_linking();
        let config = StrongDualityConfig {
            add_mccormick_for_bilinear: false,
            ..Default::default()
        };
        let pass = StrongDualityPass::new(config);
        let result = pass.apply(&prob).unwrap();

        assert_eq!(result.num_dual_vars, 1);
        assert_eq!(result.num_bilinear_terms, 0);
        assert!(!result.mccormick_added);

        // Locate the strong-duality row.
        let sd_idx = result
            .milp
            .constraint_names
            .iter()
            .position(|n| n == "strong_duality")
            .expect("strong_duality constraint should exist");
        assert_eq!(result.milp.constraint_senses[sd_idx], ConstraintSense::Eq);
        assert!((result.milp.constraint_rhs[sd_idx]).abs() < 1e-12);

        // The constraint should reference y_0 (coeff 3.0) and λ_0 (coeff −10.0).
        let sd_entries: Vec<_> = result
            .milp
            .constraint_matrix
            .entries
            .iter()
            .filter(|e| e.row == sd_idx)
            .collect();
        assert_eq!(sd_entries.len(), 2);
    }

    #[test]
    fn test_full_apply_small_bilevel() {
        let prob = small_bilevel();
        let config = StrongDualityConfig::default();
        let pass = StrongDualityPass::new(config);
        let result = pass.apply(&prob).unwrap();

        // Variables: 1 x + 1 y + 1 λ + 1 McCormick aux = 4
        assert_eq!(result.milp.num_vars, 4);
        assert_eq!(result.num_dual_vars, 1);
        assert_eq!(result.num_bilinear_terms, 1);
        assert!(result.mccormick_added);

        // Constraints:
        //   1 primal feasibility
        //   1 dual feasibility
        //   1 strong duality
        //   4 McCormick envelope
        //   1 upper constraint
        //   ─────────────────
        //   8 total
        assert_eq!(result.milp.num_constraints, 8);

        // Objective: min x + y  ⟹  obj_coeffs[x_0]=1, obj_coeffs[y_0]=1.
        assert!((result.milp.obj_coeffs[0] - 1.0).abs() < 1e-12);
        assert!((result.milp.obj_coeffs[1] - 1.0).abs() < 1e-12);
        assert!((result.milp.obj_coeffs[2]).abs() < 1e-12); // λ_0
        assert!((result.milp.obj_coeffs[3]).abs() < 1e-12); // w

        // Verify constraint names.
        assert!(result
            .milp
            .constraint_names
            .contains(&"primal_feas_0".to_string()));
        assert!(result
            .milp
            .constraint_names
            .contains(&"dual_feas_0".to_string()));
        assert!(result
            .milp
            .constraint_names
            .contains(&"strong_duality".to_string()));
        assert!(result
            .milp
            .constraint_names
            .contains(&"upper_0".to_string()));
    }

    #[test]
    fn test_error_on_non_lp_lower_level() {
        let mut prob = small_bilevel();
        // Break dimensions so verify_lp_lower_level fails.
        prob.lower_a = SparseMatrix {
            rows: 1,
            cols: 999,
            entries: vec![],
        };
        let config = StrongDualityConfig {
            verify_lp_lower_level: true,
            ..Default::default()
        };
        let pass = StrongDualityPass::new(config);
        let err = pass.apply(&prob).unwrap_err();
        let msg = format!("{}", err);
        assert!(
            msg.contains("LP lower level"),
            "error should mention LP lower level, got: {}",
            msg
        );

        // With verification disabled the pass should proceed (even though
        // dimensions are inconsistent, the sparse iteration is safe).
        let config_skip = StrongDualityConfig {
            verify_lp_lower_level: false,
            ..Default::default()
        };
        let pass_skip = StrongDualityPass::new(config_skip);
        assert!(pass_skip.apply(&prob).is_ok());
    }

    #[test]
    fn test_mccormick_disabled_warning() {
        let prob = small_bilevel();
        let config = StrongDualityConfig {
            add_mccormick_for_bilinear: false,
            ..Default::default()
        };
        let pass = StrongDualityPass::new(config);
        let result = pass.apply(&prob).unwrap();

        // No McCormick aux vars ⟹ only 3 base variables.
        assert_eq!(result.milp.num_vars, 3);
        assert!(!result.mccormick_added);

        // A warning should have been emitted about the bilinear terms.
        assert!(
            result.warnings.iter().any(|w| w.contains("bilinear")),
            "expected a bilinear-term warning, got: {:?}",
            result.warnings
        );
    }
}
