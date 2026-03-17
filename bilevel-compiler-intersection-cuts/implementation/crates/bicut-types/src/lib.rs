//! Shared types for the BiCut bilevel optimization compiler.

pub mod certificate;
pub mod config;
pub mod constraint;
pub mod error;
pub mod expression;
pub mod ir;
pub mod matrix;
pub mod problem;
pub mod signature;
pub mod solution;
pub mod variable;

use nalgebra::DMatrix;
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Tolerance for floating-point comparisons.
pub const DEFAULT_TOLERANCE: f64 = 1e-8;

/// Index type for variables.
pub type VarIndex = usize;

/// Index type for constraints.
pub type ConstraintIndex = usize;

/// A sparse matrix entry (row, col, value).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseEntry {
    pub row: usize,
    pub col: usize,
    pub value: f64,
}

/// Sparse matrix in COO format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseMatrix {
    pub rows: usize,
    pub cols: usize,
    pub entries: Vec<SparseEntry>,
}

impl SparseMatrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            entries: Vec::new(),
        }
    }

    pub fn add_entry(&mut self, row: usize, col: usize, value: f64) {
        self.entries.push(SparseEntry { row, col, value });
    }

    pub fn to_dense(&self) -> DMatrix<f64> {
        let mut mat = DMatrix::zeros(self.rows, self.cols);
        for e in &self.entries {
            mat[(e.row, e.col)] += e.value;
        }
        mat
    }

    pub fn from_dense(mat: &DMatrix<f64>) -> Self {
        let mut sparse = Self::new(mat.nrows(), mat.ncols());
        for i in 0..mat.nrows() {
            for j in 0..mat.ncols() {
                let v = mat[(i, j)];
                if v.abs() > DEFAULT_TOLERANCE {
                    sparse.add_entry(i, j, v);
                }
            }
        }
        sparse
    }
}

/// Constraint sense.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConstraintSense {
    Le,
    Ge,
    Eq,
}

/// Optimization direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptDirection {
    Minimize,
    Maximize,
}

/// Variable bound.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct VarBound {
    pub lower: f64,
    pub upper: f64,
}

impl Default for VarBound {
    fn default() -> Self {
        Self {
            lower: 0.0,
            upper: f64::INFINITY,
        }
    }
}

/// LP problem data: min c^T y  s.t.  Ay <= b, y >= 0.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LpProblem {
    pub direction: OptDirection,
    pub c: Vec<f64>,
    pub a_matrix: SparseMatrix,
    pub b_rhs: Vec<f64>,
    pub senses: Vec<ConstraintSense>,
    pub var_bounds: Vec<VarBound>,
    pub num_vars: usize,
    pub num_constraints: usize,
}

impl LpProblem {
    pub fn new(num_vars: usize, num_constraints: usize) -> Self {
        Self {
            direction: OptDirection::Minimize,
            c: vec![0.0; num_vars],
            a_matrix: SparseMatrix::new(num_constraints, num_vars),
            b_rhs: vec![0.0; num_constraints],
            senses: vec![ConstraintSense::Le; num_constraints],
            var_bounds: vec![VarBound::default(); num_vars],
            num_vars,
            num_constraints,
        }
    }
}

/// LP solution status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LpStatus {
    Optimal,
    Infeasible,
    Unbounded,
    IterationLimit,
    Unknown,
}

impl fmt::Display for LpStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LpStatus::Optimal => write!(f, "Optimal"),
            LpStatus::Infeasible => write!(f, "Infeasible"),
            LpStatus::Unbounded => write!(f, "Unbounded"),
            LpStatus::IterationLimit => write!(f, "IterationLimit"),
            LpStatus::Unknown => write!(f, "Unknown"),
        }
    }
}

/// LP solution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LpSolution {
    pub status: LpStatus,
    pub objective: f64,
    pub primal: Vec<f64>,
    pub dual: Vec<f64>,
    pub basis: Vec<BasisStatus>,
    pub iterations: u64,
}

impl LpSolution {
    pub fn infeasible() -> Self {
        Self {
            status: LpStatus::Infeasible,
            objective: f64::INFINITY,
            primal: Vec::new(),
            dual: Vec::new(),
            basis: Vec::new(),
            iterations: 0,
        }
    }

    pub fn unbounded() -> Self {
        Self {
            status: LpStatus::Unbounded,
            objective: f64::NEG_INFINITY,
            primal: Vec::new(),
            dual: Vec::new(),
            basis: Vec::new(),
            iterations: 0,
        }
    }
}

/// Basis status for a variable or constraint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BasisStatus {
    Basic,
    NonBasicLower,
    NonBasicUpper,
    SuperBasic,
}

/// A bilevel optimization problem.
/// min_{x,y} F(x, y)
/// s.t. G(x,y) <= 0  (upper-level constraints)
///      y in argmin_{y'} { c^T y' : Ay' <= b + Bx, y' >= 0 }
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BilevelProblem {
    pub upper_obj_c_x: Vec<f64>,
    pub upper_obj_c_y: Vec<f64>,
    pub lower_obj_c: Vec<f64>,
    pub lower_a: SparseMatrix,
    pub lower_b: Vec<f64>,
    pub lower_linking_b: SparseMatrix,
    pub upper_constraints_a: SparseMatrix,
    pub upper_constraints_b: Vec<f64>,
    pub num_upper_vars: usize,
    pub num_lower_vars: usize,
    pub num_lower_constraints: usize,
    pub num_upper_constraints: usize,
}

impl BilevelProblem {
    pub fn num_upper_vars(&self) -> usize {
        self.num_upper_vars
    }

    pub fn num_lower_vars(&self) -> usize {
        self.num_lower_vars
    }

    /// Build the lower-level LP for a given x.
    pub fn lower_level_lp(&self, x: &[f64]) -> LpProblem {
        let m = self.num_lower_constraints;
        let n = self.num_lower_vars;
        let mut lp = LpProblem::new(n, m);
        lp.direction = OptDirection::Minimize;
        lp.c = self.lower_obj_c.clone();
        lp.a_matrix = self.lower_a.clone();

        // b + Bx
        let mut rhs = self.lower_b.clone();
        for entry in &self.lower_linking_b.entries {
            if entry.col < x.len() {
                rhs[entry.row] += entry.value * x[entry.col];
            }
        }
        lp.b_rhs = rhs;
        lp.senses = vec![ConstraintSense::Le; m];
        lp.var_bounds = vec![VarBound::default(); n];
        lp
    }
}

/// Ordered float alias for use in maps.
pub type OrdF64 = OrderedFloat<f64>;

/// Convert a slice to ordered floats for hashing.
pub fn to_ordered(v: &[f64]) -> Vec<OrdF64> {
    v.iter().map(|&x| OrderedFloat(x)).collect()
}

/// A hyperplane representation: a^T x <= b.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Halfspace {
    pub normal: Vec<f64>,
    pub rhs: f64,
}

impl Halfspace {
    pub fn contains(&self, point: &[f64], tol: f64) -> bool {
        let dot: f64 = self
            .normal
            .iter()
            .zip(point.iter())
            .map(|(a, x)| a * x)
            .sum();
        dot <= self.rhs + tol
    }
}

/// A convex polyhedron defined by Ax <= b.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Polyhedron {
    pub halfspaces: Vec<Halfspace>,
    pub dim: usize,
}

impl Polyhedron {
    pub fn new(dim: usize) -> Self {
        Self {
            halfspaces: Vec::new(),
            dim,
        }
    }

    pub fn add_halfspace(&mut self, normal: Vec<f64>, rhs: f64) {
        self.halfspaces.push(Halfspace { normal, rhs });
    }

    pub fn contains(&self, point: &[f64], tol: f64) -> bool {
        self.halfspaces.iter().all(|h| h.contains(point, tol))
    }

    pub fn is_empty_heuristic(&self, tol: f64) -> bool {
        // Simple heuristic: try the origin
        if self.contains(&vec![0.0; self.dim], tol) {
            return false;
        }
        // Try to find any feasible point would require LP; for now return false
        false
    }
}

/// Affine function: f(x) = a^T x + b.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AffineFunction {
    pub coefficients: Vec<f64>,
    pub constant: f64,
}

impl AffineFunction {
    pub fn evaluate(&self, x: &[f64]) -> f64 {
        let dot: f64 = self
            .coefficients
            .iter()
            .zip(x.iter())
            .map(|(a, xi)| a * xi)
            .sum();
        dot + self.constant
    }

    pub fn zero(dim: usize) -> Self {
        Self {
            coefficients: vec![0.0; dim],
            constant: 0.0,
        }
    }
}

/// A valid inequality: alpha^T x + beta^T y >= gamma.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidInequality {
    pub alpha: Vec<f64>,
    pub beta: Vec<f64>,
    pub gamma: f64,
}

impl ValidInequality {
    pub fn is_violated_by(&self, x: &[f64], y: &[f64], tol: f64) -> bool {
        let lhs: f64 = self
            .alpha
            .iter()
            .zip(x.iter())
            .map(|(a, xi)| a * xi)
            .sum::<f64>()
            + self
                .beta
                .iter()
                .zip(y.iter())
                .map(|(b, yi)| b * yi)
                .sum::<f64>();
        lhs < self.gamma - tol
    }
}

// ---------------------------------------------------------------------------
// Variable classification types
// ---------------------------------------------------------------------------

/// Type of a decision variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VariableType {
    Continuous,
    Binary,
    Integer,
}

/// Scope: does the variable belong to the leader or follower?
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VariableScope {
    Leader,
    Follower,
}

// ---------------------------------------------------------------------------
// Linear expression (symbolic)
// ---------------------------------------------------------------------------

/// Sparse linear expression: Σ coeff_i * x_{var_index_i} + constant.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LinearExpr {
    pub coeffs: Vec<(usize, f64)>,
    pub constant: f64,
}

impl LinearExpr {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_coeffs(coeffs: Vec<(usize, f64)>, constant: f64) -> Self {
        Self { coeffs, constant }
    }

    pub fn add_term(&mut self, var_index: usize, coeff: f64) {
        if coeff.abs() > 1e-15 {
            self.coeffs.push((var_index, coeff));
        }
    }

    pub fn num_terms(&self) -> usize {
        self.coeffs.len()
    }

    pub fn is_zero(&self) -> bool {
        self.constant.abs() < 1e-15 && self.coeffs.iter().all(|(_, c)| c.abs() < 1e-15)
    }

    pub fn evaluate(&self, x: &[f64]) -> f64 {
        let mut val = self.constant;
        for &(idx, c) in &self.coeffs {
            if idx < x.len() {
                val += c * x[idx];
            }
        }
        val
    }

    pub fn variable_indices(&self) -> Vec<usize> {
        self.coeffs.iter().map(|&(i, _)| i).collect()
    }

    pub fn max_var_index(&self) -> Option<usize> {
        self.coeffs.iter().map(|&(i, _)| i).max()
    }
}

/// A linear constraint: expr sense rhs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearConstraint {
    pub expr: LinearExpr,
    pub sense: ConstraintSense,
    pub rhs: f64,
    pub name: Option<String>,
}

impl LinearConstraint {
    pub fn new(expr: LinearExpr, sense: ConstraintSense, rhs: f64) -> Self {
        Self {
            expr,
            sense,
            rhs,
            name: None,
        }
    }

    pub fn named(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn is_satisfied(&self, x: &[f64], tol: f64) -> bool {
        let val = self.expr.evaluate(x);
        match self.sense {
            ConstraintSense::Le => val <= self.rhs + tol,
            ConstraintSense::Ge => val >= self.rhs - tol,
            ConstraintSense::Eq => (val - self.rhs).abs() <= tol,
        }
    }

    pub fn slack(&self, x: &[f64]) -> f64 {
        let val = self.expr.evaluate(x);
        match self.sense {
            ConstraintSense::Le => self.rhs - val,
            ConstraintSense::Ge => val - self.rhs,
            ConstraintSense::Eq => (self.rhs - val).abs(),
        }
    }
}

// ---------------------------------------------------------------------------
// Lower-level type / coupling type / problem classification
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LowerLevelType {
    LP,
    QP,
    MILP,
    MIQP,
    ConvexNLP,
    GeneralNLP,
}

impl fmt::Display for LowerLevelType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LP => write!(f, "LP"),
            Self::QP => write!(f, "QP"),
            Self::MILP => write!(f, "MILP"),
            Self::MIQP => write!(f, "MIQP"),
            Self::ConvexNLP => write!(f, "ConvexNLP"),
            Self::GeneralNLP => write!(f, "GeneralNLP"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CouplingType {
    None,
    ObjectiveOnly,
    ConstraintOnly,
    Both,
}

impl fmt::Display for CouplingType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => write!(f, "None"),
            Self::ObjectiveOnly => write!(f, "ObjectiveOnly"),
            Self::ConstraintOnly => write!(f, "ConstraintOnly"),
            Self::Both => write!(f, "Both"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DifficultyClass {
    Easy,
    Moderate,
    Hard,
    VeryHard,
    Intractable,
}

impl fmt::Display for DifficultyClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Easy => write!(f, "Easy"),
            Self::Moderate => write!(f, "Moderate"),
            Self::Hard => write!(f, "Hard"),
            Self::VeryHard => write!(f, "VeryHard"),
            Self::Intractable => write!(f, "Intractable"),
        }
    }
}

/// Compact problem signature for classification and strategy selection.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ProblemSignature {
    pub lower_type: LowerLevelType,
    pub coupling_type: CouplingType,
    pub num_leader_vars: usize,
    pub num_follower_vars: usize,
    pub num_upper_constraints: usize,
    pub num_lower_constraints: usize,
    pub num_coupling_constraints: usize,
    pub has_integer_upper: bool,
    pub has_integer_lower: bool,
}

impl ProblemSignature {
    pub fn total_vars(&self) -> usize {
        self.num_leader_vars + self.num_follower_vars
    }

    pub fn total_constraints(&self) -> usize {
        self.num_upper_constraints + self.num_lower_constraints + self.num_coupling_constraints
    }

    pub fn is_pure_linear(&self) -> bool {
        matches!(self.lower_type, LowerLevelType::LP)
            && !self.has_integer_upper
            && !self.has_integer_lower
    }

    pub fn has_integers(&self) -> bool {
        self.has_integer_upper || self.has_integer_lower
    }
}

// ---------------------------------------------------------------------------
// CQ status and certificates
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CqStatus {
    Satisfied,
    Violated,
    Unknown,
    NotApplicable,
}

impl fmt::Display for CqStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Satisfied => write!(f, "Satisfied"),
            Self::Violated => write!(f, "Violated"),
            Self::Unknown => write!(f, "Unknown"),
            Self::NotApplicable => write!(f, "N/A"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CqResult {
    pub cq_name: String,
    pub status: CqStatus,
    pub confidence: f64,
    pub justification: String,
}

impl CqResult {
    pub fn satisfied(name: impl Into<String>, justification: impl Into<String>) -> Self {
        Self {
            cq_name: name.into(),
            status: CqStatus::Satisfied,
            confidence: 1.0,
            justification: justification.into(),
        }
    }

    pub fn violated(name: impl Into<String>, justification: impl Into<String>) -> Self {
        Self {
            cq_name: name.into(),
            status: CqStatus::Violated,
            confidence: 1.0,
            justification: justification.into(),
        }
    }

    pub fn unknown(
        name: impl Into<String>,
        confidence: f64,
        justification: impl Into<String>,
    ) -> Self {
        Self {
            cq_name: name.into(),
            status: CqStatus::Unknown,
            confidence,
            justification: justification.into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertificateCheck {
    pub name: String,
    pub passed: bool,
    pub details: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Certificate {
    pub id: String,
    pub problem_name: String,
    pub timestamp: String,
    pub fingerprint: String,
    pub checks: Vec<CertificateCheck>,
}

impl Certificate {
    pub fn new(problem_name: impl Into<String>, fingerprint: impl Into<String>) -> Self {
        let id = uuid::Uuid::new_v4().to_string();
        Self {
            id,
            problem_name: problem_name.into(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            fingerprint: fingerprint.into(),
            checks: Vec::new(),
        }
    }

    pub fn add_check(&mut self, name: impl Into<String>, passed: bool, details: impl Into<String>) {
        self.checks.push(CertificateCheck {
            name: name.into(),
            passed,
            details: details.into(),
        });
    }

    pub fn all_passed(&self) -> bool {
        self.checks.iter().all(|c| c.passed)
    }

    pub fn num_passed(&self) -> usize {
        self.checks.iter().filter(|c| c.passed).count()
    }

    pub fn num_failed(&self) -> usize {
        self.checks.iter().filter(|c| !c.passed).count()
    }
}

// ---------------------------------------------------------------------------
// Sparse CSR matrix
// ---------------------------------------------------------------------------

/// Compressed sparse row matrix.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseMatrixCsr {
    pub nrows: usize,
    pub ncols: usize,
    pub row_offsets: Vec<usize>,
    pub col_indices: Vec<usize>,
    pub values: Vec<f64>,
}

impl SparseMatrixCsr {
    pub fn zeros(nrows: usize, ncols: usize) -> Self {
        Self {
            nrows,
            ncols,
            row_offsets: vec![0; nrows + 1],
            col_indices: Vec::new(),
            values: Vec::new(),
        }
    }

    pub fn from_triplets(nrows: usize, ncols: usize, triplets: &[(usize, usize, f64)]) -> Self {
        let mut row_counts = vec![0usize; nrows];
        for &(r, _, _) in triplets {
            if r < nrows {
                row_counts[r] += 1;
            }
        }
        let mut row_offsets = vec![0usize; nrows + 1];
        for i in 0..nrows {
            row_offsets[i + 1] = row_offsets[i] + row_counts[i];
        }

        let nnz = row_offsets[nrows];
        let mut col_indices = vec![0usize; nnz];
        let mut values = vec![0.0f64; nnz];
        let mut pos = row_offsets.clone();

        for &(r, c, v) in triplets {
            if r < nrows {
                let idx = pos[r];
                col_indices[idx] = c;
                values[idx] = v;
                pos[r] += 1;
            }
        }

        // Sort columns within each row
        for r in 0..nrows {
            let start = row_offsets[r];
            let end = row_offsets[r + 1];
            let mut pairs: Vec<(usize, f64)> =
                (start..end).map(|i| (col_indices[i], values[i])).collect();
            pairs.sort_by_key(|&(c, _)| c);
            for (i, (c, v)) in pairs.into_iter().enumerate() {
                col_indices[start + i] = c;
                values[start + i] = v;
            }
        }

        Self {
            nrows,
            ncols,
            row_offsets,
            col_indices,
            values,
        }
    }

    pub fn from_sparse_matrix(sm: &SparseMatrix) -> Self {
        let triplets: Vec<(usize, usize, f64)> =
            sm.entries.iter().map(|e| (e.row, e.col, e.value)).collect();
        Self::from_triplets(sm.rows, sm.cols, &triplets)
    }

    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    pub fn density(&self) -> f64 {
        if self.nrows == 0 || self.ncols == 0 {
            return 0.0;
        }
        self.nnz() as f64 / (self.nrows * self.ncols) as f64
    }

    pub fn row_entries(&self, r: usize) -> Vec<(usize, f64)> {
        if r >= self.nrows {
            return Vec::new();
        }
        let start = self.row_offsets[r];
        let end = self.row_offsets[r + 1];
        (start..end)
            .map(|i| (self.col_indices[i], self.values[i]))
            .collect()
    }

    pub fn get(&self, r: usize, c: usize) -> f64 {
        if r >= self.nrows {
            return 0.0;
        }
        let start = self.row_offsets[r];
        let end = self.row_offsets[r + 1];
        for i in start..end {
            if self.col_indices[i] == c {
                return self.values[i];
            }
            if self.col_indices[i] > c {
                break;
            }
        }
        0.0
    }

    pub fn mul_vec(&self, x: &[f64]) -> Vec<f64> {
        let mut y = vec![0.0; self.nrows];
        for r in 0..self.nrows {
            let start = self.row_offsets[r];
            let end = self.row_offsets[r + 1];
            for i in start..end {
                if self.col_indices[i] < x.len() {
                    y[r] += self.values[i] * x[self.col_indices[i]];
                }
            }
        }
        y
    }

    pub fn transpose(&self) -> Self {
        let mut triplets = Vec::with_capacity(self.nnz());
        for r in 0..self.nrows {
            let start = self.row_offsets[r];
            let end = self.row_offsets[r + 1];
            for i in start..end {
                triplets.push((self.col_indices[i], r, self.values[i]));
            }
        }
        triplets.sort_by_key(|&(r, c, _)| (r, c));
        Self::from_triplets(self.ncols, self.nrows, &triplets)
    }
}

// ---------------------------------------------------------------------------
// Reformulation kind
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ReformulationKind {
    KKT,
    StrongDuality,
    ValueFunction,
    ColumnConstraintGeneration,
    BendersDecomposition,
    Regularization,
}

impl fmt::Display for ReformulationKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::KKT => write!(f, "KKT"),
            Self::StrongDuality => write!(f, "StrongDuality"),
            Self::ValueFunction => write!(f, "ValueFunction"),
            Self::ColumnConstraintGeneration => write!(f, "CCG"),
            Self::BendersDecomposition => write!(f, "Benders"),
            Self::Regularization => write!(f, "Regularization"),
        }
    }
}

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum BiCutError {
    #[error("Validation error: {0}")]
    Validation(String),
    #[error("Infeasible: {0}")]
    Infeasible(String),
    #[error("Unbounded: {0}")]
    Unbounded(String),
    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),
    #[error("Unsupported: {0}")]
    Unsupported(String),
    #[error("Internal error: {0}")]
    Internal(String),
}

pub type BiCutResult<T> = Result<T, BiCutError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_matrix_roundtrip() {
        let mut sm = SparseMatrix::new(2, 3);
        sm.add_entry(0, 1, 2.5);
        sm.add_entry(1, 0, -1.0);
        let dense = sm.to_dense();
        assert!((dense[(0, 1)] - 2.5).abs() < 1e-12);
        assert!((dense[(1, 0)] + 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_polyhedron_contains() {
        let mut p = Polyhedron::new(2);
        p.add_halfspace(vec![1.0, 0.0], 1.0);
        p.add_halfspace(vec![0.0, 1.0], 1.0);
        assert!(p.contains(&[0.5, 0.5], 1e-8));
        assert!(!p.contains(&[2.0, 0.5], 1e-8));
    }

    #[test]
    fn test_affine_function() {
        let f = AffineFunction {
            coefficients: vec![2.0, 3.0],
            constant: 1.0,
        };
        assert!((f.evaluate(&[1.0, 1.0]) - 6.0).abs() < 1e-12);
    }
}
