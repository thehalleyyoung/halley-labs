//! Structural analysis of bilevel optimization programs.
//!
//! Detects lower-level type (LP/QP/MILP), identifies coupling variables,
//! classifies constraint structure, computes problem dimensions, and
//! detects special structures (network, totally unimodular, etc.).

use bicut_types::{
    BilevelProblem, ConstraintSense, CouplingType, LowerLevelType, ProblemSignature, SparseMatrix,
    SparseMatrixCsr, DEFAULT_TOLERANCE,
};
use indexmap::IndexSet;
use log::debug;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Results of structural analysis on a bilevel problem.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuralReport {
    pub lower_level_type: LowerLevelType,
    pub coupling_type: CouplingType,
    pub signature: ProblemSignature,
    pub dimensions: ProblemDimensions,
    pub special_structures: Vec<SpecialStructure>,
    pub constraint_classification: ConstraintClassification,
    pub sparsity: SparsityInfo,
}

/// Problem dimensions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProblemDimensions {
    pub num_leader_vars: usize,
    pub num_follower_vars: usize,
    pub num_upper_constraints: usize,
    pub num_lower_constraints: usize,
    pub num_coupling_constraints: usize,
    pub total_vars: usize,
    pub total_constraints: usize,
}

/// Detected special structures.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SpecialStructure {
    NetworkFlow,
    TotallyUnimodular,
    BoxConstrained,
    BoundedVariables,
    SingleFollowerConstraint,
    DiagonalLowerLevel,
    BlockDiagonal,
    RankDeficient,
    EmptyUpperConstraints,
    EmptyLowerLinking,
    IdentityLowerA,
    PurelyLinearCoupling,
}

/// Classification of constraints in the problem.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintClassification {
    pub num_equality: usize,
    pub num_inequality_le: usize,
    pub num_inequality_ge: usize,
    pub num_bound_constraints: usize,
    pub num_range_constraints: usize,
    pub num_dense_rows: usize,
    pub num_singleton_rows: usize,
    pub max_row_nnz: usize,
    pub avg_row_nnz: f64,
}

/// Sparsity information for the problem matrices.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparsityInfo {
    pub lower_a_density: f64,
    pub lower_a_nnz: usize,
    pub linking_density: f64,
    pub linking_nnz: usize,
    pub upper_density: f64,
    pub upper_nnz: usize,
}

// ---------------------------------------------------------------------------
// Analyzer
// ---------------------------------------------------------------------------

/// Main entry point for structural analysis.
pub struct StructuralAnalysis;

impl StructuralAnalysis {
    /// Perform complete structural analysis on a bilevel problem.
    pub fn analyze(problem: &BilevelProblem) -> StructuralReport {
        let dimensions = Self::compute_dimensions(problem);
        let lower_level_type = Self::detect_lower_level_type(problem);
        let coupling_type = Self::detect_coupling_type(problem);
        let constraint_classification = Self::classify_constraints(problem);
        let special_structures = Self::detect_special_structures(problem);
        let sparsity = Self::compute_sparsity(problem);

        let signature = ProblemSignature {
            lower_type: lower_level_type,
            coupling_type,
            num_leader_vars: dimensions.num_leader_vars,
            num_follower_vars: dimensions.num_follower_vars,
            num_upper_constraints: dimensions.num_upper_constraints,
            num_lower_constraints: dimensions.num_lower_constraints,
            num_coupling_constraints: dimensions.num_coupling_constraints,
            has_integer_upper: false,
            has_integer_lower: false,
        };

        debug!(
            "Structural analysis complete: lower={}, coupling={}",
            lower_level_type, coupling_type
        );

        StructuralReport {
            lower_level_type,
            coupling_type,
            signature,
            dimensions,
            special_structures,
            constraint_classification,
            sparsity,
        }
    }

    /// Compute problem dimensions from the raw BilevelProblem.
    pub fn compute_dimensions(problem: &BilevelProblem) -> ProblemDimensions {
        let nl = problem.num_upper_vars;
        let nf = problem.num_lower_vars;
        let mc = problem.num_upper_constraints;
        let ml = problem.num_lower_constraints;
        // Coupling constraints are rows in the linking matrix that are nonzero
        let coupling = count_nonzero_rows(&problem.lower_linking_b);
        ProblemDimensions {
            num_leader_vars: nl,
            num_follower_vars: nf,
            num_upper_constraints: mc,
            num_lower_constraints: ml,
            num_coupling_constraints: coupling,
            total_vars: nl + nf,
            total_constraints: mc + ml,
        }
    }

    /// Detect the type of the lower-level problem.
    pub fn detect_lower_level_type(problem: &BilevelProblem) -> LowerLevelType {
        // The lower level is: min c^T y  s.t. Ay <= b + Bx
        // All variables continuous, linear objective, linear constraints → LP
        // Check if lower_obj_c is all-zero with quadratic terms → QP (not present in current model)
        // Since BilevelProblem stores only linear data, the lower level is LP by default.
        // We check for integer variable indicators via variable bounds.
        let has_binary_hint = detect_binary_pattern(&problem.lower_obj_c);
        if has_binary_hint {
            LowerLevelType::MILP
        } else {
            LowerLevelType::LP
        }
    }

    /// Detect how leader and follower are coupled.
    pub fn detect_coupling_type(problem: &BilevelProblem) -> CouplingType {
        let obj_coupling = has_objective_coupling(problem);
        let constraint_coupling = has_constraint_coupling(problem);

        match (obj_coupling, constraint_coupling) {
            (false, false) => CouplingType::None,
            (true, false) => CouplingType::ObjectiveOnly,
            (false, true) => CouplingType::ConstraintOnly,
            (true, true) => CouplingType::Both,
        }
    }

    /// Classify constraints by type, density, etc.
    pub fn classify_constraints(problem: &BilevelProblem) -> ConstraintClassification {
        let csr = SparseMatrixCsr::from_sparse_matrix(&problem.lower_a);
        let m = problem.num_lower_constraints;

        let mut num_singleton = 0usize;
        let mut num_dense = 0usize;
        let mut max_nnz = 0usize;
        let mut total_nnz = 0usize;

        for r in 0..csr.nrows {
            let start = csr.row_offsets[r];
            let end = csr.row_offsets[r + 1];
            let row_nnz = end - start;
            total_nnz += row_nnz;
            if row_nnz > max_nnz {
                max_nnz = row_nnz;
            }
            if row_nnz <= 1 {
                num_singleton += 1;
            }
            let n = problem.num_lower_vars;
            if n > 0 && row_nnz as f64 / n as f64 > 0.5 {
                num_dense += 1;
            }
        }

        let avg_nnz = if m > 0 {
            total_nnz as f64 / m as f64
        } else {
            0.0
        };

        // Count equality vs. inequality in lower-level (all Le by default in current model)
        let num_equality = 0;
        let num_le = m;
        let num_ge = 0;

        // Detect bound constraints: rows with a single nonzero coefficient = ±1
        let num_bound = count_bound_constraints(&csr);

        ConstraintClassification {
            num_equality,
            num_inequality_le: num_le,
            num_inequality_ge: num_ge,
            num_bound_constraints: num_bound,
            num_range_constraints: 0,
            num_dense_rows: num_dense,
            num_singleton_rows: num_singleton,
            max_row_nnz: max_nnz,
            avg_row_nnz: avg_nnz,
        }
    }

    /// Detect special structural properties.
    pub fn detect_special_structures(problem: &BilevelProblem) -> Vec<SpecialStructure> {
        let mut structures = Vec::new();
        let csr = SparseMatrixCsr::from_sparse_matrix(&problem.lower_a);

        if problem.num_upper_constraints == 0 {
            structures.push(SpecialStructure::EmptyUpperConstraints);
        }

        if problem.lower_linking_b.entries.is_empty() {
            structures.push(SpecialStructure::EmptyLowerLinking);
        }

        if check_box_constrained(&csr, problem.num_lower_vars) {
            structures.push(SpecialStructure::BoxConstrained);
        }

        if check_all_bounded(problem) {
            structures.push(SpecialStructure::BoundedVariables);
        }

        if problem.num_lower_constraints == 1 {
            structures.push(SpecialStructure::SingleFollowerConstraint);
        }

        if check_diagonal(&csr) {
            structures.push(SpecialStructure::DiagonalLowerLevel);
        }

        if check_identity(&csr) {
            structures.push(SpecialStructure::IdentityLowerA);
        }

        if check_totally_unimodular_heuristic(&csr) {
            structures.push(SpecialStructure::TotallyUnimodular);
        }

        if check_network_heuristic(&csr) {
            structures.push(SpecialStructure::NetworkFlow);
        }

        if check_block_diagonal(&csr) {
            structures.push(SpecialStructure::BlockDiagonal);
        }

        if has_rank_deficiency(&csr) {
            structures.push(SpecialStructure::RankDeficient);
        }

        let linking_csr = SparseMatrixCsr::from_sparse_matrix(&problem.lower_linking_b);
        if linking_csr.nnz() > 0 && all_coefficients_unit_or_zero(&linking_csr) {
            structures.push(SpecialStructure::PurelyLinearCoupling);
        }

        structures
    }

    /// Compute sparsity statistics.
    pub fn compute_sparsity(problem: &BilevelProblem) -> SparsityInfo {
        let lower_a_nnz = problem.lower_a.entries.len();
        let linking_nnz = problem.lower_linking_b.entries.len();
        let upper_nnz = problem.upper_constraints_a.entries.len();

        let lower_a_total = problem.num_lower_constraints * problem.num_lower_vars;
        let linking_total = problem.lower_linking_b.rows * problem.lower_linking_b.cols;
        let upper_total =
            problem.num_upper_constraints * (problem.num_upper_vars + problem.num_lower_vars);

        SparsityInfo {
            lower_a_density: safe_density(lower_a_nnz, lower_a_total),
            lower_a_nnz,
            linking_density: safe_density(linking_nnz, linking_total),
            linking_nnz,
            upper_density: safe_density(upper_nnz, upper_total),
            upper_nnz,
        }
    }

    /// Build the lower-level constraint matrix as CSR.
    pub fn lower_level_csr(problem: &BilevelProblem) -> SparseMatrixCsr {
        SparseMatrixCsr::from_sparse_matrix(&problem.lower_a)
    }

    /// Build the linking matrix as CSR.
    pub fn linking_csr(problem: &BilevelProblem) -> SparseMatrixCsr {
        SparseMatrixCsr::from_sparse_matrix(&problem.lower_linking_b)
    }

    /// Extract the set of leader variable indices that appear in lower-level constraints.
    pub fn coupling_leader_indices(problem: &BilevelProblem) -> IndexSet<usize> {
        let mut indices = IndexSet::new();
        for entry in &problem.lower_linking_b.entries {
            if entry.value.abs() > DEFAULT_TOLERANCE {
                indices.insert(entry.col);
            }
        }
        indices
    }

    /// Extract the set of follower variable indices that appear in upper-level constraints.
    pub fn coupling_follower_indices(problem: &BilevelProblem) -> IndexSet<usize> {
        let mut indices = IndexSet::new();
        // Upper constraints matrix has columns [x, y]; follower columns start at num_upper_vars
        let nx = problem.num_upper_vars;
        for entry in &problem.upper_constraints_a.entries {
            if entry.col >= nx && entry.value.abs() > DEFAULT_TOLERANCE {
                indices.insert(entry.col - nx);
            }
        }
        indices
    }

    /// Compute the ratio of nonzero linking entries to total possible entries.
    pub fn coupling_density(problem: &BilevelProblem) -> f64 {
        let total = problem.num_lower_constraints * problem.num_upper_vars;
        if total == 0 {
            return 0.0;
        }
        let nnz = problem
            .lower_linking_b
            .entries
            .iter()
            .filter(|e| e.value.abs() > DEFAULT_TOLERANCE)
            .count();
        nnz as f64 / total as f64
    }

    /// Count the maximum number of leader variables appearing in any single
    /// lower-level constraint.
    pub fn max_coupling_per_constraint(problem: &BilevelProblem) -> usize {
        let csr = SparseMatrixCsr::from_sparse_matrix(&problem.lower_linking_b);
        let mut max_count = 0usize;
        for r in 0..csr.nrows {
            let entries = csr.row_entries(r);
            let cnt = entries
                .iter()
                .filter(|(_, v)| v.abs() > DEFAULT_TOLERANCE)
                .count();
            if cnt > max_count {
                max_count = cnt;
            }
        }
        max_count
    }

    /// Compute coefficient range (max / min nonzero) for the lower-level matrix.
    pub fn coefficient_range(problem: &BilevelProblem) -> f64 {
        let abs_vals: Vec<f64> = problem
            .lower_a
            .entries
            .iter()
            .map(|e| e.value.abs())
            .filter(|&v| v > DEFAULT_TOLERANCE)
            .collect();
        if abs_vals.is_empty() {
            return 1.0;
        }
        let min_val = abs_vals.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = abs_vals.iter().cloned().fold(0.0f64, f64::max);
        if min_val < DEFAULT_TOLERANCE {
            return f64::INFINITY;
        }
        max_val / min_val
    }

    /// Build a hash map from constraint row index to the set of variable indices
    /// that appear in that row.
    pub fn constraint_variable_map(problem: &BilevelProblem) -> HashMap<usize, Vec<usize>> {
        let csr = SparseMatrixCsr::from_sparse_matrix(&problem.lower_a);
        let mut map = HashMap::new();
        for r in 0..csr.nrows {
            let cols: Vec<usize> = csr
                .row_entries(r)
                .iter()
                .filter(|(_, v)| v.abs() > DEFAULT_TOLERANCE)
                .map(|(c, _)| *c)
                .collect();
            map.insert(r, cols);
        }
        map
    }

    /// Build a hash map from variable index to the set of constraint row indices
    /// that contain it.
    pub fn variable_constraint_map(problem: &BilevelProblem) -> HashMap<usize, Vec<usize>> {
        let mut map: HashMap<usize, Vec<usize>> = HashMap::new();
        for entry in &problem.lower_a.entries {
            if entry.value.abs() > DEFAULT_TOLERANCE {
                map.entry(entry.col).or_default().push(entry.row);
            }
        }
        // Sort the constraint indices
        for v in map.values_mut() {
            v.sort_unstable();
            v.dedup();
        }
        map
    }

    /// Check whether lower-level matrix has full row rank (heuristic).
    pub fn has_full_row_rank(problem: &BilevelProblem) -> bool {
        let m = problem.num_lower_constraints;
        let n = problem.num_lower_vars;
        if m > n {
            return false;
        }
        let csr = SparseMatrixCsr::from_sparse_matrix(&problem.lower_a);
        // Heuristic: check that each row has at least one nonzero and no two rows
        // have identical support.
        let mut supports: Vec<Vec<usize>> = Vec::new();
        for r in 0..m {
            let entries = csr.row_entries(r);
            if entries.is_empty() {
                return false;
            }
            let sup: Vec<usize> = entries.iter().map(|(c, _)| *c).collect();
            supports.push(sup);
        }
        let mut seen: HashSet<Vec<usize>> = HashSet::new();
        for sup in &supports {
            if !seen.insert(sup.clone()) {
                return false;
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn count_nonzero_rows(sm: &SparseMatrix) -> usize {
    let mut rows: HashSet<usize> = HashSet::new();
    for entry in &sm.entries {
        if entry.value.abs() > DEFAULT_TOLERANCE {
            rows.insert(entry.row);
        }
    }
    rows.len()
}

fn has_objective_coupling(problem: &BilevelProblem) -> bool {
    // Upper objective contains both x and y terms
    let has_x = problem
        .upper_obj_c_x
        .iter()
        .any(|&c| c.abs() > DEFAULT_TOLERANCE);
    let has_y = problem
        .upper_obj_c_y
        .iter()
        .any(|&c| c.abs() > DEFAULT_TOLERANCE);
    has_x && has_y
}

fn has_constraint_coupling(problem: &BilevelProblem) -> bool {
    // Lower-level constraints depend on x via the linking matrix
    !problem.lower_linking_b.entries.is_empty()
        && problem
            .lower_linking_b
            .entries
            .iter()
            .any(|e| e.value.abs() > DEFAULT_TOLERANCE)
}

fn detect_binary_pattern(coeffs: &[f64]) -> bool {
    // Heuristic: if all coefficients are 0 or 1, might indicate binary variables
    // This is a rough heuristic; a real system would have explicit type annotations
    if coeffs.is_empty() {
        return false;
    }
    let all_zero_one = coeffs
        .iter()
        .all(|&c| c.abs() < DEFAULT_TOLERANCE || (c - 1.0).abs() < DEFAULT_TOLERANCE);
    // Only flag if there are enough 1-coefficients
    let num_ones = coeffs
        .iter()
        .filter(|&&c| (c - 1.0).abs() < DEFAULT_TOLERANCE)
        .count();
    all_zero_one && num_ones > 0 && num_ones == coeffs.len()
}

fn safe_density(nnz: usize, total: usize) -> f64 {
    if total == 0 {
        0.0
    } else {
        nnz as f64 / total as f64
    }
}

fn count_bound_constraints(csr: &SparseMatrixCsr) -> usize {
    let mut count = 0;
    for r in 0..csr.nrows {
        let entries = csr.row_entries(r);
        if entries.len() == 1 {
            let (_, v) = entries[0];
            if (v.abs() - 1.0).abs() < DEFAULT_TOLERANCE {
                count += 1;
            }
        }
    }
    count
}

fn check_box_constrained(csr: &SparseMatrixCsr, num_vars: usize) -> bool {
    // Box constrained = each row has exactly one nonzero with |coeff| = 1
    if csr.nrows == 0 || csr.nrows < num_vars {
        return false;
    }
    for r in 0..csr.nrows {
        let entries = csr.row_entries(r);
        if entries.len() != 1 {
            return false;
        }
        if (entries[0].1.abs() - 1.0).abs() > DEFAULT_TOLERANCE {
            return false;
        }
    }
    true
}

fn check_all_bounded(problem: &BilevelProblem) -> bool {
    // All lower-level variable bounds are finite
    problem
        .lower_level_lp(&vec![0.0; problem.num_upper_vars])
        .var_bounds
        .iter()
        .all(|b| b.lower.is_finite() && b.upper.is_finite())
}

fn check_diagonal(csr: &SparseMatrixCsr) -> bool {
    if csr.nrows == 0 || csr.nrows != csr.ncols {
        return false;
    }
    for r in 0..csr.nrows {
        let entries = csr.row_entries(r);
        if entries.len() != 1 || entries[0].0 != r {
            return false;
        }
    }
    true
}

fn check_identity(csr: &SparseMatrixCsr) -> bool {
    if !check_diagonal(csr) {
        return false;
    }
    for r in 0..csr.nrows {
        let entries = csr.row_entries(r);
        if (entries[0].1 - 1.0).abs() > DEFAULT_TOLERANCE {
            return false;
        }
    }
    true
}

fn check_totally_unimodular_heuristic(csr: &SparseMatrixCsr) -> bool {
    // TU heuristic: all nonzero entries are ±1 and each column has at most
    // one +1 and one −1 entry.
    let mut col_plus: HashMap<usize, usize> = HashMap::new();
    let mut col_minus: HashMap<usize, usize> = HashMap::new();

    for r in 0..csr.nrows {
        for (c, v) in csr.row_entries(r) {
            if (v - 1.0).abs() < DEFAULT_TOLERANCE {
                *col_plus.entry(c).or_insert(0) += 1;
            } else if (v + 1.0).abs() < DEFAULT_TOLERANCE {
                *col_minus.entry(c).or_insert(0) += 1;
            } else {
                return false;
            }
        }
    }

    for (&col, &cnt) in &col_plus {
        if cnt > 1 {
            return false;
        }
    }
    for (&col, &cnt) in &col_minus {
        if cnt > 1 {
            return false;
        }
    }
    true
}

fn check_network_heuristic(csr: &SparseMatrixCsr) -> bool {
    // Network flow: each column has exactly one +1 and one -1 (node-arc incidence).
    if csr.nnz() == 0 {
        return false;
    }
    let mut col_sums: HashMap<usize, (usize, usize)> = HashMap::new();
    for r in 0..csr.nrows {
        for (c, v) in csr.row_entries(r) {
            let entry = col_sums.entry(c).or_insert((0, 0));
            if (v - 1.0).abs() < DEFAULT_TOLERANCE {
                entry.0 += 1;
            } else if (v + 1.0).abs() < DEFAULT_TOLERANCE {
                entry.1 += 1;
            } else {
                return false;
            }
        }
    }
    col_sums.values().all(|&(p, m)| p == 1 && m == 1)
}

fn check_block_diagonal(csr: &SparseMatrixCsr) -> bool {
    // Check if the matrix decomposes into independent blocks via simple
    // connected-component analysis on a row-column bipartite graph.
    if csr.nrows <= 1 {
        return false;
    }
    let mut uf = UnionFind::new(csr.nrows + csr.ncols);
    for r in 0..csr.nrows {
        for (c, v) in csr.row_entries(r) {
            if v.abs() > DEFAULT_TOLERANCE {
                uf.union(r, csr.nrows + c);
            }
        }
    }
    let components: HashSet<usize> = (0..csr.nrows).map(|r| uf.find(r)).collect();
    components.len() > 1
}

fn has_rank_deficiency(csr: &SparseMatrixCsr) -> bool {
    // Heuristic: check for duplicate rows.
    let mut row_hashes: HashMap<Vec<(usize, ordered_float::OrderedFloat<f64>)>, usize> =
        HashMap::new();
    for r in 0..csr.nrows {
        let entries: Vec<(usize, ordered_float::OrderedFloat<f64>)> = csr
            .row_entries(r)
            .iter()
            .map(|&(c, v)| (c, ordered_float::OrderedFloat(v)))
            .collect();
        if row_hashes.contains_key(&entries) {
            return true;
        }
        row_hashes.insert(entries, r);
    }
    false
}

fn all_coefficients_unit_or_zero(csr: &SparseMatrixCsr) -> bool {
    for r in 0..csr.nrows {
        for (_, v) in csr.row_entries(r) {
            if v.abs() > DEFAULT_TOLERANCE && (v.abs() - 1.0).abs() > DEFAULT_TOLERANCE {
                return false;
            }
        }
    }
    true
}

// Simple union-find for block diagonal detection
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }

    fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        if self.rank[ra] < self.rank[rb] {
            self.parent[ra] = rb;
        } else if self.rank[ra] > self.rank[rb] {
            self.parent[rb] = ra;
        } else {
            self.parent[rb] = ra;
            self.rank[ra] += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use bicut_types::{BilevelProblem, SparseMatrix, VarBound};

    fn make_simple_problem() -> BilevelProblem {
        // min_x  x + y
        // s.t.   x >= 0
        //        y in argmin { -y : y <= 1 + x, y >= 0 }
        let mut lower_a = SparseMatrix::new(1, 1);
        lower_a.add_entry(0, 0, 1.0); // y <= ...

        let mut linking = SparseMatrix::new(1, 1);
        linking.add_entry(0, 0, 1.0); // ... + 1*x

        let upper_a = SparseMatrix::new(0, 2);

        BilevelProblem {
            upper_obj_c_x: vec![1.0],
            upper_obj_c_y: vec![1.0],
            lower_obj_c: vec![-1.0],
            lower_a,
            lower_b: vec![1.0],
            lower_linking_b: linking,
            upper_constraints_a: upper_a,
            upper_constraints_b: vec![],
            num_upper_vars: 1,
            num_lower_vars: 1,
            num_lower_constraints: 1,
            num_upper_constraints: 0,
        }
    }

    fn make_decoupled_problem() -> BilevelProblem {
        let mut lower_a = SparseMatrix::new(2, 2);
        lower_a.add_entry(0, 0, 1.0);
        lower_a.add_entry(1, 1, 1.0);

        let linking = SparseMatrix::new(2, 1);
        let upper_a = SparseMatrix::new(0, 3);

        BilevelProblem {
            upper_obj_c_x: vec![1.0],
            upper_obj_c_y: vec![0.0, 0.0],
            lower_obj_c: vec![1.0, 1.0],
            lower_a,
            lower_b: vec![5.0, 5.0],
            lower_linking_b: linking,
            upper_constraints_a: upper_a,
            upper_constraints_b: vec![],
            num_upper_vars: 1,
            num_lower_vars: 2,
            num_lower_constraints: 2,
            num_upper_constraints: 0,
        }
    }

    #[test]
    fn test_dimensions() {
        let p = make_simple_problem();
        let dims = StructuralAnalysis::compute_dimensions(&p);
        assert_eq!(dims.num_leader_vars, 1);
        assert_eq!(dims.num_follower_vars, 1);
        assert_eq!(dims.total_vars, 2);
    }

    #[test]
    fn test_lower_level_type_lp() {
        let p = make_simple_problem();
        let lt = StructuralAnalysis::detect_lower_level_type(&p);
        assert_eq!(lt, LowerLevelType::LP);
    }

    #[test]
    fn test_coupling_type_both() {
        let p = make_simple_problem();
        let ct = StructuralAnalysis::detect_coupling_type(&p);
        assert_eq!(ct, CouplingType::Both);
    }

    #[test]
    fn test_coupling_type_none() {
        let p = make_decoupled_problem();
        let ct = StructuralAnalysis::detect_coupling_type(&p);
        // upper_obj has c_x=[1] and c_y=[0,0], so only x in objective
        // linking is empty, so no constraint coupling
        assert_eq!(ct, CouplingType::None);
    }

    #[test]
    fn test_empty_upper_structure() {
        let p = make_simple_problem();
        let structs = StructuralAnalysis::detect_special_structures(&p);
        assert!(structs.contains(&SpecialStructure::EmptyUpperConstraints));
    }

    #[test]
    fn test_diagonal_detection() {
        let p = make_decoupled_problem();
        let structs = StructuralAnalysis::detect_special_structures(&p);
        assert!(structs.contains(&SpecialStructure::DiagonalLowerLevel));
    }

    #[test]
    fn test_full_analysis() {
        let p = make_simple_problem();
        let report = StructuralAnalysis::analyze(&p);
        assert_eq!(report.lower_level_type, LowerLevelType::LP);
        assert_eq!(report.signature.num_leader_vars, 1);
    }

    #[test]
    fn test_coefficient_range() {
        let p = make_simple_problem();
        let range = StructuralAnalysis::coefficient_range(&p);
        assert!((range - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_coupling_leader_indices() {
        let p = make_simple_problem();
        let indices = StructuralAnalysis::coupling_leader_indices(&p);
        assert!(indices.contains(&0));
    }

    #[test]
    fn test_sparsity_info() {
        let p = make_simple_problem();
        let sp = StructuralAnalysis::compute_sparsity(&p);
        assert_eq!(sp.lower_a_nnz, 1);
        assert_eq!(sp.linking_nnz, 1);
    }
}
