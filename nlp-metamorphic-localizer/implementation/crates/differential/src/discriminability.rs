//! N1 Stage Discriminability Matrix: determines whether the transformation
//! suite can distinguish all pipeline stages.

use serde::{Deserialize, Serialize};
use shared_types::{
    IntermediateRepresentation, LocalizerError, Result, StageId, TransformationId,
};
use std::collections::{HashMap, HashSet};

use crate::stage_differential::DifferentialComputer;

// ── DiscriminabilityMatrix ──────────────────────────────────────────────────

/// The N1 matrix M ∈ ℝ^{n×m} where M_{k,j} = E[Δ_k(x, τ_j)].
/// Rows = stages (n), columns = transformations (m).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscriminabilityMatrix {
    pub matrix: Vec<Vec<f64>>,
    pub stage_names: Vec<StageId>,
    pub transformation_names: Vec<TransformationId>,
    pub num_stages: usize,
    pub num_transformations: usize,
}

impl DiscriminabilityMatrix {
    /// Construct from precomputed matrix data.
    pub fn new(
        matrix: Vec<Vec<f64>>,
        stage_names: Vec<StageId>,
        transformation_names: Vec<TransformationId>,
    ) -> Result<Self> {
        let num_stages = stage_names.len();
        let num_transformations = transformation_names.len();
        if matrix.len() != num_stages {
            return Err(LocalizerError::validation("validation", format!(
                "Matrix has {} rows but {} stage names",
                matrix.len(),
                num_stages
            )));
        }
        for (i, row) in matrix.iter().enumerate() {
            if row.len() != num_transformations {
                return Err(LocalizerError::validation("validation", format!(
                    "Row {} has {} cols, expected {}",
                    i,
                    row.len(),
                    num_transformations
                )));
            }
        }
        Ok(Self {
            matrix,
            stage_names,
            transformation_names,
            num_stages,
            num_transformations,
        })
    }

    /// Build the discriminability matrix from calibration data.
    ///
    /// `samples[stage_idx][transform_idx]` = list of differential values
    /// from calibration runs. Computes E[Δ_k(x, τ_j)] as the mean.
    pub fn from_calibration_samples(
        samples: &[Vec<Vec<f64>>],
        stage_names: Vec<StageId>,
        transformation_names: Vec<TransformationId>,
    ) -> Result<Self> {
        let num_stages = stage_names.len();
        let num_transformations = transformation_names.len();

        if samples.len() != num_stages {
            return Err(LocalizerError::validation("validation", format!(
                "samples has {} stage entries, expected {}",
                samples.len(),
                num_stages
            )));
        }

        let mut matrix = Vec::with_capacity(num_stages);
        for (k, stage_samples) in samples.iter().enumerate() {
            if stage_samples.len() != num_transformations {
                return Err(LocalizerError::validation("validation", format!(
                    "Stage {} has {} transformation entries, expected {}",
                    k,
                    stage_samples.len(),
                    num_transformations
                )));
            }
            let row: Vec<f64> = stage_samples
                .iter()
                .map(|vals| {
                    if vals.is_empty() {
                        0.0
                    } else {
                        vals.iter().sum::<f64>() / vals.len() as f64
                    }
                })
                .collect();
            matrix.push(row);
        }

        Self::new(matrix, stage_names, transformation_names)
    }

    /// Compute the discriminability matrix from original/transformed IR pairs.
    ///
    /// For each stage k and transformation j, computes the mean differential
    /// across all provided calibration samples.
    pub fn compute_discriminability_matrix(
        computer: &DifferentialComputer,
        stage_ids: &[StageId],
        transformation_ids: &[TransformationId],
        calibration_pairs: &[Vec<Vec<(IntermediateRepresentation, IntermediateRepresentation)>>],
    ) -> Result<Self> {
        let num_stages = stage_ids.len();
        let num_transforms = transformation_ids.len();

        if calibration_pairs.len() != num_stages {
            return Err(LocalizerError::validation(
                "validation",
                "calibration_pairs outer dimension must match num_stages",
            ));
        }

        let mut matrix = Vec::with_capacity(num_stages);
        for (k, stage_pairs) in calibration_pairs.iter().enumerate() {
            if stage_pairs.len() != num_transforms {
                return Err(LocalizerError::validation("validation", format!(
                    "Stage {} has {} transform entries, expected {}",
                    k,
                    stage_pairs.len(),
                    num_transforms
                )));
            }
            let mut row = Vec::with_capacity(num_transforms);
            for pairs in stage_pairs {
                if pairs.is_empty() {
                    row.push(0.0);
                    continue;
                }
                let mut sum = 0.0;
                for (orig, trans) in pairs {
                    let diff =
                        computer.compute_stage_differential(&stage_ids[k], k, orig, trans)?;
                    sum += diff.delta_value;
                }
                row.push(sum / pairs.len() as f64);
            }
            matrix.push(row);
        }

        Self::new(
            matrix,
            stage_ids.to_vec(),
            transformation_ids.to_vec(),
        )
    }

    pub fn get(&self, stage: usize, transform: usize) -> Option<f64> {
        self.matrix
            .get(stage)
            .and_then(|row| row.get(transform).copied())
    }

    /// Compute the numerical rank using Gaussian elimination with partial pivoting.
    pub fn compute_rank(&self) -> usize {
        let ge = GaussianElimination::new(&self.matrix);
        ge.rank()
    }

    /// Compute the condition number (ratio of largest to smallest singular value).
    pub fn compute_condition_number(&self) -> f64 {
        let svd = SingularValueDecomposition::compute(&self.matrix);
        if svd.singular_values.is_empty() {
            return f64::INFINITY;
        }
        let max_sv = svd
            .singular_values
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let min_sv = svd
            .singular_values
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        if min_sv < 1e-15 {
            f64::INFINITY
        } else {
            max_sv / min_sv
        }
    }

    /// Whether the matrix has full row rank (rank == n_stages).
    pub fn is_full_rank(&self) -> bool {
        self.compute_rank() >= self.num_stages
    }

    /// When rank < n, identify sets of stages that cannot be distinguished.
    pub fn find_indistinguishable_stages(&self) -> Vec<Vec<StageId>> {
        let tolerance = 1e-8;
        let mut groups: Vec<Vec<usize>> = Vec::new();
        let mut assigned: HashSet<usize> = HashSet::new();

        for i in 0..self.num_stages {
            if assigned.contains(&i) {
                continue;
            }
            let mut group = vec![i];
            assigned.insert(i);
            for j in (i + 1)..self.num_stages {
                if assigned.contains(&j) {
                    continue;
                }
                // Two stages are indistinguishable if their rows are (nearly) proportional
                if rows_proportional(&self.matrix[i], &self.matrix[j], tolerance) {
                    group.push(j);
                    assigned.insert(j);
                }
            }
            if group.len() > 1 {
                groups.push(group);
            }
        }

        groups
            .into_iter()
            .map(|g| g.into_iter().map(|i| self.stage_names[i].clone()).collect())
            .collect()
    }

    /// Suggest transformation types to add to improve coverage.
    pub fn suggest_transformations(
        &self,
        undercovered_stages: &[StageId],
    ) -> Vec<String> {
        let mut suggestions = Vec::new();
        for stage in undercovered_stages {
            if let Some(idx) = self.stage_names.iter().position(|s| s == stage) {
                let row = &self.matrix[idx];
                let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                if max_val < 0.01 {
                    suggestions.push(format!(
                        "Stage '{}' has near-zero response to all transformations; \
                         add a transformation targeting its specific IR type",
                        stage
                    ));
                } else {
                    // Find which transformations this stage responds to least
                    let mut indexed: Vec<(usize, f64)> =
                        row.iter().copied().enumerate().collect();
                    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                    let weakest = &indexed[0];
                    suggestions.push(format!(
                        "Stage '{}' responds weakly to transformation '{}' (Δ={:.4}); \
                         consider a complementary transformation",
                        stage, self.transformation_names[weakest.0], weakest.1
                    ));
                }
            }
        }
        if suggestions.is_empty() {
            suggestions.push(
                "Add transformations that produce diverse IR-type-specific effects".into(),
            );
        }
        suggestions
    }

    /// Produce a full discriminability report.
    pub fn report(&self) -> DiscriminabilityReport {
        let rank = self.compute_rank();
        let condition_number = self.compute_condition_number();
        let indistinguishable_pairs = self.find_indistinguishable_stages();
        let coverage_gaps = self.find_coverage_gaps();
        let suggestions = if !coverage_gaps.is_empty() {
            self.suggest_transformations(&coverage_gaps)
        } else if !indistinguishable_pairs.is_empty() {
            let stages: Vec<StageId> = indistinguishable_pairs
                .iter()
                .flatten()
                .cloned()
                .collect();
            self.suggest_transformations(&stages)
        } else {
            Vec::new()
        };

        DiscriminabilityReport {
            rank,
            condition_number,
            indistinguishable_pairs,
            coverage_gaps,
            suggestions,
        }
    }

    fn find_coverage_gaps(&self) -> Vec<StageId> {
        let mut gaps = Vec::new();
        for (i, row) in self.matrix.iter().enumerate() {
            let max_response = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            if max_response < 0.01 {
                gaps.push(self.stage_names[i].clone());
            }
        }
        gaps
    }

    /// Validate that calibration data is sufficient.
    pub fn validate_calibration_data(samples: &[Vec<Vec<f64>>], min_samples: usize) -> Result<()> {
        for (k, stage_samples) in samples.iter().enumerate() {
            for (j, transform_samples) in stage_samples.iter().enumerate() {
                if transform_samples.len() < min_samples {
                    return Err(LocalizerError::calibration(
                        format!(
                            "Stage {} transform {} has {} samples, need at least {}",
                            k,
                            j,
                            transform_samples.len(),
                            min_samples
                        ),
                        0.0,
                    ));
                }
                // Check for variance
                if transform_samples.len() > 1 {
                    let mean =
                        transform_samples.iter().sum::<f64>() / transform_samples.len() as f64;
                    let var = transform_samples
                        .iter()
                        .map(|v| (v - mean).powi(2))
                        .sum::<f64>()
                        / (transform_samples.len() - 1) as f64;
                    if var < 1e-15 {
                        return Err(LocalizerError::calibration(
                            format!(
                                "Stage {} transform {} has zero variance (all identical samples)",
                                k, j
                            ),
                            0.0,
                        ));
                    }
                }
            }
        }
        Ok(())
    }
}

// ── DiscriminabilityReport ──────────────────────────────────────────────────

/// Summary report of discriminability analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscriminabilityReport {
    pub rank: usize,
    pub condition_number: f64,
    pub indistinguishable_pairs: Vec<Vec<StageId>>,
    pub coverage_gaps: Vec<StageId>,
    pub suggestions: Vec<String>,
}

impl DiscriminabilityReport {
    pub fn is_healthy(&self) -> bool {
        self.coverage_gaps.is_empty()
            && self.indistinguishable_pairs.is_empty()
            && self.condition_number < 1e6
    }
}

// ── GaussianElimination ─────────────────────────────────────────────────────

/// Gaussian elimination with partial pivoting for rank computation.
pub struct GaussianElimination {
    matrix: Vec<Vec<f64>>,
    rows: usize,
    cols: usize,
}

impl GaussianElimination {
    pub fn new(matrix: &[Vec<f64>]) -> Self {
        let rows = matrix.len();
        let cols = if rows > 0 { matrix[0].len() } else { 0 };
        let matrix: Vec<Vec<f64>> = matrix.to_vec();
        Self { matrix, rows, cols }
    }

    /// Perform elimination and return the rank (number of non-zero pivots).
    pub fn rank(&self) -> usize {
        let mut m = self.matrix.clone();
        let tolerance = 1e-10;
        let mut pivot_row = 0;

        for col in 0..self.cols {
            if pivot_row >= self.rows {
                break;
            }
            // Find best pivot
            let mut max_val = 0.0f64;
            let mut max_row = pivot_row;
            for row in pivot_row..self.rows {
                let val = m[row][col].abs();
                if val > max_val {
                    max_val = val;
                    max_row = row;
                }
            }
            if max_val < tolerance {
                continue;
            }
            // Swap rows
            m.swap(pivot_row, max_row);
            // Eliminate below
            let pivot_val = m[pivot_row][col];
            for row in (pivot_row + 1)..self.rows {
                let factor = m[row][col] / pivot_val;
                for c in col..self.cols {
                    let v = m[pivot_row][c];
                    m[row][c] -= factor * v;
                }
            }
            pivot_row += 1;
        }
        pivot_row
    }

    /// Row echelon form.
    pub fn row_echelon(&self) -> Vec<Vec<f64>> {
        let mut m = self.matrix.clone();
        let tolerance = 1e-10;
        let mut pivot_row = 0;

        for col in 0..self.cols {
            if pivot_row >= self.rows {
                break;
            }
            let mut max_val = 0.0f64;
            let mut max_row = pivot_row;
            for row in pivot_row..self.rows {
                let val = m[row][col].abs();
                if val > max_val {
                    max_val = val;
                    max_row = row;
                }
            }
            if max_val < tolerance {
                continue;
            }
            m.swap(pivot_row, max_row);
            let pivot_val = m[pivot_row][col];
            for c in col..self.cols {
                m[pivot_row][c] /= pivot_val;
            }
            for row in (pivot_row + 1)..self.rows {
                let factor = m[row][col];
                for c in col..self.cols {
                    let v = m[pivot_row][c];
                    m[row][c] -= factor * v;
                }
            }
            pivot_row += 1;
        }
        m
    }
}

// ── SingularValueDecomposition ──────────────────────────────────────────────

/// Simple SVD approximation using the power method on A^T A.
pub struct SingularValueDecomposition {
    pub singular_values: Vec<f64>,
}

impl SingularValueDecomposition {
    /// Compute singular values of the given matrix.
    pub fn compute(matrix: &[Vec<f64>]) -> Self {
        let rows = matrix.len();
        if rows == 0 {
            return Self {
                singular_values: Vec::new(),
            };
        }
        let cols = matrix[0].len();

        // Compute A^T A (cols × cols)
        let ata = mat_transpose_times_mat(matrix);
        let dim = ata.len();

        // Find eigenvalues of A^T A via iterative deflation + power method
        let mut deflated = ata;
        let mut singular_values = Vec::new();
        let max_rank = rows.min(cols);

        for _ in 0..max_rank {
            let eigvec = power_iteration_for_svd(&deflated, 300);
            let eigenvalue = rayleigh_quotient_vec(&deflated, &eigvec);

            if eigenvalue < 1e-20 {
                break;
            }
            singular_values.push(eigenvalue.sqrt());

            // Deflate
            for i in 0..dim {
                for j in 0..dim {
                    deflated[i][j] -= eigenvalue * eigvec[i] * eigvec[j];
                }
            }
        }

        singular_values.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        Self { singular_values }
    }

    pub fn largest(&self) -> f64 {
        self.singular_values.first().copied().unwrap_or(0.0)
    }

    pub fn smallest(&self) -> f64 {
        self.singular_values.last().copied().unwrap_or(0.0)
    }

    pub fn condition_number(&self) -> f64 {
        let s = self.smallest();
        if s < 1e-15 {
            f64::INFINITY
        } else {
            self.largest() / s
        }
    }
}

// ── Helper functions ────────────────────────────────────────────────────────

fn mat_transpose_times_mat(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let rows = a.len();
    let cols = if rows > 0 { a[0].len() } else { 0 };
    let mut result = vec![vec![0.0; cols]; cols];
    for k in 0..rows {
        for i in 0..cols {
            for j in i..cols {
                let v = a[k][i] * a[k][j];
                result[i][j] += v;
                if i != j {
                    result[j][i] += v;
                }
            }
        }
    }
    result
}

fn power_iteration_for_svd(matrix: &[Vec<f64>], max_iters: usize) -> Vec<f64> {
    let n = matrix.len();
    if n == 0 {
        return Vec::new();
    }
    // Start with [1, 1, ...] normalized
    let norm0 = (n as f64).sqrt();
    let mut v: Vec<f64> = vec![1.0 / norm0; n];

    for _ in 0..max_iters {
        let mut new_v = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                new_v[i] += matrix[i][j] * v[j];
            }
        }
        let norm: f64 = new_v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-15 {
            return v;
        }
        for x in &mut new_v {
            *x /= norm;
        }
        // Check convergence
        let diff: f64 = v
            .iter()
            .zip(new_v.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        v = new_v;
        if diff < 1e-12 {
            break;
        }
    }
    v
}

fn rayleigh_quotient_vec(matrix: &[Vec<f64>], v: &[f64]) -> f64 {
    let n = v.len();
    let mut av = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            av[i] += matrix[i][j] * v[j];
        }
    }
    let vtav: f64 = v.iter().zip(av.iter()).map(|(a, b)| a * b).sum();
    let vtv: f64 = v.iter().map(|x| x * x).sum();
    if vtv < 1e-15 {
        0.0
    } else {
        vtav / vtv
    }
}

fn rows_proportional(a: &[f64], b: &[f64], tolerance: f64) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();

    if norm_a < tolerance && norm_b < tolerance {
        return true; // both zero rows
    }
    if norm_a < tolerance || norm_b < tolerance {
        return false; // one zero, one not
    }

    // Normalize and check if a/||a|| ≈ ±b/||b||
    let a_norm: Vec<f64> = a.iter().map(|x| x / norm_a).collect();
    let b_norm: Vec<f64> = b.iter().map(|x| x / norm_b).collect();

    let dot: f64 = a_norm.iter().zip(b_norm.iter()).map(|(x, y)| x * y).sum();
    (dot.abs() - 1.0).abs() < tolerance
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_dm() -> DiscriminabilityMatrix {
        // 3 stages × 4 transformations
        let matrix = vec![
            vec![0.5, 0.1, 0.3, 0.2],
            vec![0.1, 0.6, 0.2, 0.4],
            vec![0.2, 0.3, 0.7, 0.1],
        ];
        let stages = vec![
            StageId::new("tok"),
            StageId::new("pos"),
            StageId::new("ner"),
        ];
        let transforms = vec![
            TransformationId::new("syn"),
            TransformationId::new("neg"),
            TransformationId::new("pass"),
            TransformationId::new("agr"),
        ];
        DiscriminabilityMatrix::new(matrix, stages, transforms).unwrap()
    }

    #[test]
    fn test_construction() {
        let dm = make_dm();
        assert_eq!(dm.num_stages, 3);
        assert_eq!(dm.num_transformations, 4);
        assert_eq!(dm.get(0, 0), Some(0.5));
    }

    #[test]
    fn test_full_rank_matrix() {
        let dm = make_dm();
        let rank = dm.compute_rank();
        assert_eq!(rank, 3);
        assert!(dm.is_full_rank());
    }

    #[test]
    fn test_rank_deficient_matrix() {
        // Row 2 = 2 × Row 0 → rank should be 2
        let matrix = vec![
            vec![1.0, 2.0, 3.0],
            vec![0.0, 1.0, 0.0],
            vec![2.0, 4.0, 6.0],
        ];
        let stages = vec![
            StageId::new("s0"),
            StageId::new("s1"),
            StageId::new("s2"),
        ];
        let transforms = vec![
            TransformationId::new("t0"),
            TransformationId::new("t1"),
            TransformationId::new("t2"),
        ];
        let dm = DiscriminabilityMatrix::new(matrix, stages, transforms).unwrap();
        assert_eq!(dm.compute_rank(), 2);
        assert!(!dm.is_full_rank());
    }

    #[test]
    fn test_indistinguishable_stages() {
        let matrix = vec![
            vec![1.0, 2.0],
            vec![2.0, 4.0], // proportional to row 0
            vec![0.5, 0.3],
        ];
        let stages = vec![
            StageId::new("s0"),
            StageId::new("s1"),
            StageId::new("s2"),
        ];
        let transforms = vec![
            TransformationId::new("t0"),
            TransformationId::new("t1"),
        ];
        let dm = DiscriminabilityMatrix::new(matrix, stages, transforms).unwrap();
        let groups = dm.find_indistinguishable_stages();
        assert_eq!(groups.len(), 1);
        assert!(groups[0].contains(&StageId::new("s0")));
        assert!(groups[0].contains(&StageId::new("s1")));
    }

    #[test]
    fn test_condition_number_well_conditioned() {
        let dm = make_dm();
        let cond = dm.compute_condition_number();
        assert!(cond > 1.0);
        assert!(cond < 100.0); // well-conditioned
    }

    #[test]
    fn test_condition_number_ill_conditioned() {
        let matrix = vec![
            vec![1.0, 1.0],
            vec![1.0, 1.0 + 1e-12],
        ];
        let dm = DiscriminabilityMatrix::new(
            matrix,
            vec![StageId::new("s0"), StageId::new("s1")],
            vec![TransformationId::new("t0"), TransformationId::new("t1")],
        )
        .unwrap();
        let cond = dm.compute_condition_number();
        assert!(cond > 1e6);
    }

    #[test]
    fn test_from_calibration_samples() {
        // 2 stages, 2 transforms, 3 samples each
        let samples = vec![
            vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]],
            vec![vec![0.7, 0.8, 0.9], vec![0.2, 0.3, 0.4]],
        ];
        let stages = vec![StageId::new("s0"), StageId::new("s1")];
        let transforms = vec![TransformationId::new("t0"), TransformationId::new("t1")];
        let dm =
            DiscriminabilityMatrix::from_calibration_samples(&samples, stages, transforms)
                .unwrap();
        assert_eq!(dm.num_stages, 2);
        assert!((dm.get(0, 0).unwrap() - 0.2).abs() < 1e-9);
        assert!((dm.get(1, 0).unwrap() - 0.8).abs() < 1e-9);
    }

    #[test]
    fn test_suggest_transformations() {
        let matrix = vec![
            vec![0.5, 0.3],
            vec![0.001, 0.002], // near zero → coverage gap
        ];
        let stages = vec![StageId::new("tok"), StageId::new("ner")];
        let transforms = vec![TransformationId::new("syn"), TransformationId::new("neg")];
        let dm = DiscriminabilityMatrix::new(matrix, stages, transforms).unwrap();
        let suggestions = dm.suggest_transformations(&[StageId::new("ner")]);
        assert!(!suggestions.is_empty());
        assert!(suggestions[0].contains("ner"));
    }

    #[test]
    fn test_report_healthy() {
        let dm = make_dm();
        let report = dm.report();
        assert_eq!(report.rank, 3);
        assert!(report.is_healthy());
        assert!(report.coverage_gaps.is_empty());
    }

    #[test]
    fn test_validate_calibration_insufficient() {
        let samples = vec![vec![vec![0.1], vec![0.2, 0.3]]]; // first has only 1 sample
        let result = DiscriminabilityMatrix::validate_calibration_data(&samples, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_calibration_zero_variance() {
        let samples = vec![vec![vec![0.5, 0.5, 0.5]]]; // zero variance
        let result = DiscriminabilityMatrix::validate_calibration_data(&samples, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_gaussian_elimination_rank() {
        let m = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let ge = GaussianElimination::new(&m);
        assert_eq!(ge.rank(), 3);
    }

    #[test]
    fn test_gaussian_elimination_rank_deficient() {
        let m = vec![
            vec![1.0, 2.0],
            vec![2.0, 4.0], // row 1 = 2 * row 0
        ];
        let ge = GaussianElimination::new(&m);
        assert_eq!(ge.rank(), 1);
    }

    #[test]
    fn test_row_echelon() {
        let m = vec![vec![2.0, 4.0], vec![1.0, 3.0]];
        let ge = GaussianElimination::new(&m);
        let ref_form = ge.row_echelon();
        assert!((ref_form[0][0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_svd_identity() {
        let m = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let svd = SingularValueDecomposition::compute(&m);
        assert_eq!(svd.singular_values.len(), 3);
        for sv in &svd.singular_values {
            assert!((sv - 1.0).abs() < 0.1);
        }
    }

    #[test]
    fn test_svd_known_values() {
        // Diagonal matrix with known singular values
        let m = vec![
            vec![3.0, 0.0],
            vec![0.0, 2.0],
        ];
        let svd = SingularValueDecomposition::compute(&m);
        assert!((svd.largest() - 3.0).abs() < 0.1);
        assert!((svd.smallest() - 2.0).abs() < 0.1);
        assert!((svd.condition_number() - 1.5).abs() < 0.1);
    }

    #[test]
    fn test_rows_proportional_yes() {
        assert!(rows_proportional(&[1.0, 2.0, 3.0], &[2.0, 4.0, 6.0], 1e-8));
    }

    #[test]
    fn test_rows_proportional_no() {
        assert!(!rows_proportional(&[1.0, 2.0, 3.0], &[1.0, 1.0, 1.0], 1e-8));
    }
}
