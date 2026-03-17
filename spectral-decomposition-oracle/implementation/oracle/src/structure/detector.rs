// Structure type classification for MIP instances.
// Detects block-angular, bordered-block-diagonal, staircase, network, and unstructured patterns.

use crate::error::{OracleError, OracleResult};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Types of decomposable structure in a MIP.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StructureType {
    BlockAngular,
    BorderedBlockDiagonal,
    Staircase,
    Network,
    Unstructured,
}

impl StructureType {
    pub fn all() -> &'static [StructureType] {
        &[
            StructureType::BlockAngular,
            StructureType::BorderedBlockDiagonal,
            StructureType::Staircase,
            StructureType::Network,
            StructureType::Unstructured,
        ]
    }
}

impl std::fmt::Display for StructureType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StructureType::BlockAngular => write!(f, "BlockAngular"),
            StructureType::BorderedBlockDiagonal => write!(f, "BorderedBlockDiagonal"),
            StructureType::Staircase => write!(f, "Staircase"),
            StructureType::Network => write!(f, "Network"),
            StructureType::Unstructured => write!(f, "Unstructured"),
        }
    }
}

/// Confidence score for a detected structure type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructureScore {
    pub structure_type: StructureType,
    pub score: f64,
    pub details: HashMap<String, f64>,
}

/// Result of structure detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionResult {
    pub ranked_structures: Vec<StructureScore>,
    pub partition: Vec<Vec<usize>>,
    pub linking_constraints: Vec<usize>,
    pub linking_variables: Vec<usize>,
    pub n_blocks: usize,
    pub block_sizes: Vec<usize>,
}

impl DetectionResult {
    pub fn best_structure(&self) -> Option<&StructureScore> {
        self.ranked_structures.first()
    }

    pub fn is_decomposable(&self) -> bool {
        self.ranked_structures
            .first()
            .map(|s| s.structure_type != StructureType::Unstructured && s.score > 0.3)
            .unwrap_or(false)
    }
}

/// Sparse representation of the constraint matrix.
#[derive(Debug, Clone)]
pub struct SparseMatrix {
    pub n_rows: usize,
    pub n_cols: usize,
    /// row_indices[row] = set of column indices with nonzeros
    pub row_to_cols: Vec<HashSet<usize>>,
    /// col_indices[col] = set of row indices with nonzeros
    pub col_to_rows: Vec<HashSet<usize>>,
}

impl SparseMatrix {
    pub fn new(n_rows: usize, n_cols: usize) -> Self {
        Self {
            n_rows,
            n_cols,
            row_to_cols: vec![HashSet::new(); n_rows],
            col_to_rows: vec![HashSet::new(); n_cols],
        }
    }

    pub fn add_entry(&mut self, row: usize, col: usize) {
        if row < self.n_rows && col < self.n_cols {
            self.row_to_cols[row].insert(col);
            self.col_to_rows[col].insert(row);
        }
    }

    pub fn nnz(&self) -> usize {
        self.row_to_cols.iter().map(|r| r.len()).sum()
    }

    pub fn density(&self) -> f64 {
        let total = self.n_rows * self.n_cols;
        if total == 0 {
            return 0.0;
        }
        self.nnz() as f64 / total as f64
    }
}

/// Structure detector for MIP instances.
#[derive(Debug)]
pub struct StructureDetector {
    pub min_block_size: usize,
    pub max_linking_fraction: f64,
}

impl StructureDetector {
    pub fn new() -> Self {
        Self {
            min_block_size: 2,
            max_linking_fraction: 0.3,
        }
    }

    /// Detect structure in a constraint matrix given a partition of constraints.
    pub fn detect(
        &self,
        matrix: &SparseMatrix,
        partition: &[Vec<usize>],
    ) -> OracleResult<DetectionResult> {
        if partition.is_empty() {
            return Err(OracleError::invalid_input("empty partition"));
        }

        let n_blocks = partition.len();
        let block_sizes: Vec<usize> = partition.iter().map(|b| b.len()).collect();

        // Identify linking constraints and variables
        let (linking_constraints, linking_variables) =
            self.find_linking_elements(matrix, partition);

        // Score each structure type
        let mut scores = Vec::new();

        scores.push(self.score_block_angular(matrix, partition, &linking_constraints, &linking_variables));
        scores.push(self.score_bordered_block_diagonal(matrix, partition, &linking_constraints, &linking_variables));
        scores.push(self.score_staircase(matrix, partition));
        scores.push(self.score_network(matrix));
        scores.push(self.score_unstructured(matrix, partition, &linking_constraints, &linking_variables));

        // Sort by score descending
        scores.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(DetectionResult {
            ranked_structures: scores,
            partition: partition.to_vec(),
            linking_constraints,
            linking_variables,
            n_blocks,
            block_sizes,
        })
    }

    /// Find constraints/variables that span multiple blocks.
    fn find_linking_elements(
        &self,
        matrix: &SparseMatrix,
        partition: &[Vec<usize>],
    ) -> (Vec<usize>, Vec<usize>) {
        // Build row-to-block mapping
        let mut row_to_block = vec![usize::MAX; matrix.n_rows];
        for (block_idx, block) in partition.iter().enumerate() {
            for &row in block {
                if row < row_to_block.len() {
                    row_to_block[row] = block_idx;
                }
            }
        }

        // Linking constraints: constraints not assigned to a block or spanning multiple
        let linking_constraints: Vec<usize> = (0..matrix.n_rows)
            .filter(|&r| row_to_block[r] == usize::MAX)
            .collect();

        // Build column-to-blocks mapping
        let mut col_blocks: Vec<HashSet<usize>> = vec![HashSet::new(); matrix.n_cols];
        for (block_idx, block) in partition.iter().enumerate() {
            for &row in block {
                if row < matrix.n_rows {
                    for &col in &matrix.row_to_cols[row] {
                        col_blocks[col].insert(block_idx);
                    }
                }
            }
        }

        // Linking variables: variables appearing in more than one block
        let linking_variables: Vec<usize> = (0..matrix.n_cols)
            .filter(|&c| col_blocks[c].len() > 1)
            .collect();

        (linking_constraints, linking_variables)
    }

    fn score_block_angular(
        &self,
        matrix: &SparseMatrix,
        partition: &[Vec<usize>],
        _linking_constraints: &[usize],
        linking_variables: &[usize],
    ) -> StructureScore {
        // Block-angular: independent blocks with coupling constraints.
        // Good if few linking variables and many independent blocks.
        let total_vars = matrix.n_cols;
        let linking_frac = if total_vars > 0 {
            linking_variables.len() as f64 / total_vars as f64
        } else {
            1.0
        };

        let n_blocks = partition.len();
        let block_balance = if n_blocks > 1 {
            let sizes: Vec<f64> = partition.iter().map(|b| b.len() as f64).collect();
            let mean = sizes.iter().sum::<f64>() / sizes.len() as f64;
            let cv = if mean > 0.0 {
                let var = sizes.iter().map(|&s| (s - mean).powi(2)).sum::<f64>() / sizes.len() as f64;
                var.sqrt() / mean
            } else {
                1.0
            };
            (1.0 - cv).max(0.0)
        } else {
            0.0
        };

        let score = (1.0 - linking_frac) * 0.6 + block_balance * 0.3 + if n_blocks > 1 { 0.1 } else { 0.0 };

        let mut details = HashMap::new();
        details.insert("linking_variable_fraction".to_string(), linking_frac);
        details.insert("n_blocks".to_string(), n_blocks as f64);
        details.insert("block_balance".to_string(), block_balance);

        StructureScore {
            structure_type: StructureType::BlockAngular,
            score: score.max(0.0).min(1.0),
            details,
        }
    }

    fn score_bordered_block_diagonal(
        &self,
        matrix: &SparseMatrix,
        partition: &[Vec<usize>],
        linking_constraints: &[usize],
        linking_variables: &[usize],
    ) -> StructureScore {
        // BBD: blocks + shared variables forming a border.
        let total_constraints = matrix.n_rows;
        let total_vars = matrix.n_cols;

        let constraint_linking_frac = if total_constraints > 0 {
            linking_constraints.len() as f64 / total_constraints as f64
        } else {
            1.0
        };
        let var_linking_frac = if total_vars > 0 {
            linking_variables.len() as f64 / total_vars as f64
        } else {
            1.0
        };

        // BBD is good when there are both linking constraints and variables forming a border
        let has_border = constraint_linking_frac > 0.01 && var_linking_frac > 0.01;
        let border_size = (constraint_linking_frac + var_linking_frac) / 2.0;
        let moderate_border = border_size > 0.01 && border_size < 0.3;

        let score = if has_border && moderate_border {
            (1.0 - border_size * 2.0) * 0.5 + if partition.len() > 1 { 0.3 } else { 0.0 } + 0.2
        } else {
            0.1
        };

        let mut details = HashMap::new();
        details.insert("constraint_linking_fraction".to_string(), constraint_linking_frac);
        details.insert("variable_linking_fraction".to_string(), var_linking_frac);
        details.insert("border_size".to_string(), border_size);

        StructureScore {
            structure_type: StructureType::BorderedBlockDiagonal,
            score: score.max(0.0).min(1.0),
            details,
        }
    }

    fn score_staircase(
        &self,
        matrix: &SparseMatrix,
        partition: &[Vec<usize>],
    ) -> StructureScore {
        // Staircase: sequential blocks where block k shares variables only with block k-1 and k+1.
        if partition.len() < 2 {
            return StructureScore {
                structure_type: StructureType::Staircase,
                score: 0.0,
                details: HashMap::new(),
            };
        }

        // Check sequential coupling pattern
        let n_blocks = partition.len();
        let mut block_vars: Vec<HashSet<usize>> = Vec::new();
        for block in partition {
            let mut vars = HashSet::new();
            for &row in block {
                if row < matrix.n_rows {
                    vars.extend(&matrix.row_to_cols[row]);
                }
            }
            block_vars.push(vars);
        }

        let mut sequential_overlaps = 0usize;
        let mut non_sequential_overlaps = 0usize;
        let mut total_pairs = 0usize;

        for i in 0..n_blocks {
            for j in (i + 1)..n_blocks {
                let overlap = block_vars[i].intersection(&block_vars[j]).count();
                if overlap > 0 {
                    if j == i + 1 {
                        sequential_overlaps += 1;
                    } else {
                        non_sequential_overlaps += 1;
                    }
                }
                total_pairs += 1;
            }
        }

        let staircase_score = if total_pairs > 0 {
            let sequential_frac = sequential_overlaps as f64
                / (sequential_overlaps + non_sequential_overlaps).max(1) as f64;
            sequential_frac
        } else {
            0.0
        };

        let mut details = HashMap::new();
        details.insert("sequential_overlaps".to_string(), sequential_overlaps as f64);
        details.insert("non_sequential_overlaps".to_string(), non_sequential_overlaps as f64);
        details.insert("n_blocks".to_string(), n_blocks as f64);

        StructureScore {
            structure_type: StructureType::Staircase,
            score: staircase_score.max(0.0).min(1.0),
            details,
        }
    }

    fn score_network(
        &self,
        matrix: &SparseMatrix,
    ) -> StructureScore {
        // Network flow: each column appears in at most 2 constraints with coefficients +1/-1.
        // Approximate by checking column participation.
        let mut network_cols = 0usize;
        let total_cols = matrix.n_cols;

        for col in 0..matrix.n_cols {
            let n_rows = matrix.col_to_rows[col].len();
            if n_rows <= 2 {
                network_cols += 1;
            }
        }

        let network_fraction = if total_cols > 0 {
            network_cols as f64 / total_cols as f64
        } else {
            0.0
        };

        // Also check constraint density (network matrices are sparse)
        let density_score = (1.0 - matrix.density().min(0.5) * 2.0).max(0.0);

        let score = network_fraction * 0.7 + density_score * 0.3;

        let mut details = HashMap::new();
        details.insert("network_column_fraction".to_string(), network_fraction);
        details.insert("density_score".to_string(), density_score);

        StructureScore {
            structure_type: StructureType::Network,
            score: score.max(0.0).min(1.0),
            details,
        }
    }

    fn score_unstructured(
        &self,
        matrix: &SparseMatrix,
        partition: &[Vec<usize>],
        linking_constraints: &[usize],
        linking_variables: &[usize],
    ) -> StructureScore {
        // High score when linking fraction is high and blocks are imbalanced.
        let total_vars = matrix.n_cols;
        let total_constraints = matrix.n_rows;

        let var_linking_frac = if total_vars > 0 {
            linking_variables.len() as f64 / total_vars as f64
        } else {
            1.0
        };
        let constraint_linking_frac = if total_constraints > 0 {
            linking_constraints.len() as f64 / total_constraints as f64
        } else {
            1.0
        };

        let linking_score = (var_linking_frac + constraint_linking_frac) / 2.0;

        // Single block or very imbalanced blocks
        let block_penalty = if partition.len() <= 1 {
            0.5
        } else {
            let sizes: Vec<f64> = partition.iter().map(|b| b.len() as f64).collect();
            let max_size = sizes.iter().cloned().fold(0.0_f64, f64::max);
            let total: f64 = sizes.iter().sum();
            if total > 0.0 {
                (max_size / total - 1.0 / partition.len() as f64).max(0.0) * 2.0
            } else {
                0.5
            }
        };

        let score = linking_score * 0.5 + block_penalty * 0.3 + matrix.density() * 0.2;

        let mut details = HashMap::new();
        details.insert("linking_score".to_string(), linking_score);
        details.insert("block_penalty".to_string(), block_penalty);
        details.insert("density".to_string(), matrix.density());

        StructureScore {
            structure_type: StructureType::Unstructured,
            score: score.max(0.0).min(1.0),
            details,
        }
    }

    /// Auto-detect structure from a constraint matrix without a pre-existing partition.
    pub fn auto_detect(&self, matrix: &SparseMatrix) -> OracleResult<DetectionResult> {
        // Simple greedy partitioning based on connected components of the row graph
        let partition = self.greedy_partition(matrix);
        self.detect(matrix, &partition)
    }

    /// Greedy partitioning: group rows by shared variables.
    fn greedy_partition(&self, matrix: &SparseMatrix) -> Vec<Vec<usize>> {
        let n = matrix.n_rows;
        if n == 0 {
            return vec![];
        }

        let mut visited = vec![false; n];
        let mut blocks = Vec::new();

        for start in 0..n {
            if visited[start] {
                continue;
            }
            let mut block = Vec::new();
            let mut queue = vec![start];
            visited[start] = true;

            while let Some(row) = queue.pop() {
                block.push(row);
                // Find neighbors: rows sharing a variable with this row
                for &col in &matrix.row_to_cols[row] {
                    for &neighbor in &matrix.col_to_rows[col] {
                        if !visited[neighbor] {
                            visited[neighbor] = true;
                            queue.push(neighbor);
                        }
                    }
                }
                // Limit block size to prevent one giant block
                if block.len() > n / 2 + 1 {
                    break;
                }
            }
            blocks.push(block);
        }

        blocks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_block_diagonal_matrix() -> SparseMatrix {
        // 2 independent blocks of constraints
        let mut m = SparseMatrix::new(6, 6);
        // Block 1: rows 0-2, cols 0-2
        m.add_entry(0, 0);
        m.add_entry(0, 1);
        m.add_entry(1, 0);
        m.add_entry(1, 2);
        m.add_entry(2, 1);
        m.add_entry(2, 2);
        // Block 2: rows 3-5, cols 3-5
        m.add_entry(3, 3);
        m.add_entry(3, 4);
        m.add_entry(4, 3);
        m.add_entry(4, 5);
        m.add_entry(5, 4);
        m.add_entry(5, 5);
        m
    }

    fn make_coupled_matrix() -> SparseMatrix {
        // Two blocks coupled by variable 6
        let mut m = SparseMatrix::new(6, 7);
        // Block 1
        m.add_entry(0, 0);
        m.add_entry(0, 6);
        m.add_entry(1, 1);
        m.add_entry(2, 2);
        // Block 2
        m.add_entry(3, 3);
        m.add_entry(3, 6);
        m.add_entry(4, 4);
        m.add_entry(5, 5);
        m
    }

    #[test]
    fn test_sparse_matrix_basic() {
        let m = make_block_diagonal_matrix();
        assert_eq!(m.n_rows, 6);
        assert_eq!(m.n_cols, 6);
        assert_eq!(m.nnz(), 12);
    }

    #[test]
    fn test_sparse_matrix_density() {
        let m = make_block_diagonal_matrix();
        let d = m.density();
        assert!((d - 12.0 / 36.0).abs() < 1e-10);
    }

    #[test]
    fn test_structure_type_display() {
        assert_eq!(StructureType::BlockAngular.to_string(), "BlockAngular");
        assert_eq!(StructureType::Network.to_string(), "Network");
    }

    #[test]
    fn test_detect_block_angular() {
        let m = make_block_diagonal_matrix();
        let partition = vec![vec![0, 1, 2], vec![3, 4, 5]];
        let detector = StructureDetector::new();
        let result = detector.detect(&m, &partition).unwrap();

        assert!(!result.ranked_structures.is_empty());
        assert_eq!(result.n_blocks, 2);
        assert!(result.is_decomposable());
    }

    #[test]
    fn test_detect_linking_variables() {
        let m = make_coupled_matrix();
        let partition = vec![vec![0, 1, 2], vec![3, 4, 5]];
        let detector = StructureDetector::new();
        let result = detector.detect(&m, &partition).unwrap();

        assert!(result.linking_variables.contains(&6));
    }

    #[test]
    fn test_auto_detect() {
        let m = make_block_diagonal_matrix();
        let detector = StructureDetector::new();
        let result = detector.auto_detect(&m).unwrap();
        // Should find 2 blocks
        assert!(result.n_blocks >= 1);
    }

    #[test]
    fn test_detect_empty_partition() {
        let m = make_block_diagonal_matrix();
        let detector = StructureDetector::new();
        assert!(detector.detect(&m, &[]).is_err());
    }

    #[test]
    fn test_detection_result_best() {
        let m = make_block_diagonal_matrix();
        let partition = vec![vec![0, 1, 2], vec![3, 4, 5]];
        let detector = StructureDetector::new();
        let result = detector.detect(&m, &partition).unwrap();
        assert!(result.best_structure().is_some());
    }

    #[test]
    fn test_greedy_partition() {
        let m = make_block_diagonal_matrix();
        let detector = StructureDetector::new();
        let partition = detector.greedy_partition(&m);
        // Two disconnected components
        assert!(partition.len() >= 2);
    }

    #[test]
    fn test_all_structure_types() {
        let all = StructureType::all();
        assert_eq!(all.len(), 5);
    }

    #[test]
    fn test_network_scoring() {
        let mut m = SparseMatrix::new(4, 4);
        // Each column in at most 2 rows
        m.add_entry(0, 0);
        m.add_entry(1, 0);
        m.add_entry(1, 1);
        m.add_entry(2, 1);
        m.add_entry(2, 2);
        m.add_entry(3, 2);
        m.add_entry(3, 3);

        let detector = StructureDetector::new();
        let score = detector.score_network(&m);
        assert!(score.score > 0.5);
    }
}
