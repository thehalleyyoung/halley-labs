// Dantzig-Wolfe amenability detection: identify linking constraints and score
// partitions for DW decomposition suitability.

use crate::error::{OracleError, OracleResult};
use crate::structure::detector::SparseMatrix;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// A linking constraint that spans multiple blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinkingConstraint {
    pub index: usize,
    pub n_blocks_spanned: usize,
    pub n_variables: usize,
    pub density: f64,
}

/// Score for Dantzig-Wolfe decomposition amenability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DWScore {
    pub overall_score: f64,
    pub n_linking: usize,
    pub linking_fraction: f64,
    pub block_independence: f64,
    pub avg_block_size: f64,
    pub linking_density: f64,
    pub block_sizes: Vec<usize>,
    pub linking_constraints: Vec<LinkingConstraint>,
}

impl DWScore {
    pub fn is_amenable(&self) -> bool {
        self.overall_score > 0.5
    }

    pub fn summary(&self) -> String {
        format!(
            "DW score: {:.3} | linking constraints: {} ({:.1}%) | block independence: {:.3} | avg block: {:.0}",
            self.overall_score,
            self.n_linking,
            self.linking_fraction * 100.0,
            self.block_independence,
            self.avg_block_size
        )
    }
}

/// Detector for Dantzig-Wolfe decomposition amenability.
#[derive(Debug)]
pub struct DWDetector {
    pub max_linking_fraction: f64,
    pub min_block_size: usize,
    pub min_blocks: usize,
}

impl DWDetector {
    pub fn new() -> Self {
        Self {
            max_linking_fraction: 0.2,
            min_block_size: 3,
            min_blocks: 2,
        }
    }

    /// Analyze a partition for DW amenability.
    pub fn analyze(
        &self,
        matrix: &SparseMatrix,
        partition: &[Vec<usize>],
    ) -> OracleResult<DWScore> {
        if partition.is_empty() {
            return Err(OracleError::invalid_input("empty partition"));
        }

        // Build row-to-block mapping
        let mut row_to_block = vec![None; matrix.n_rows];
        for (block_idx, block) in partition.iter().enumerate() {
            for &row in block {
                if row < matrix.n_rows {
                    row_to_block[row] = Some(block_idx);
                }
            }
        }

        // Identify linking constraints (rows not in any block or spanning multiple blocks)
        let linking = self.identify_linking_constraints(matrix, partition, &row_to_block);

        let total_constraints = matrix.n_rows;
        let linking_fraction = if total_constraints > 0 {
            linking.len() as f64 / total_constraints as f64
        } else {
            1.0
        };

        // Compute block independence
        let independence = self.compute_block_independence(matrix, partition, &linking);

        let block_sizes: Vec<usize> = partition.iter().map(|b| b.len()).collect();
        let avg_block_size = if block_sizes.is_empty() {
            0.0
        } else {
            block_sizes.iter().sum::<usize>() as f64 / block_sizes.len() as f64
        };

        // Linking constraint density
        let linking_density = self.compute_linking_density(matrix, &linking);

        // Overall score
        let few_linking = (1.0 - linking_fraction / self.max_linking_fraction)
            .max(0.0)
            .min(1.0);
        let multi_blocks = if partition.len() >= self.min_blocks {
            0.2
        } else {
            0.0
        };
        let size_bonus = if avg_block_size >= self.min_block_size as f64 {
            0.1
        } else {
            0.0
        };
        let sparse_linking = (1.0 - linking_density).max(0.0) * 0.1;

        let overall_score =
            few_linking * 0.4 + independence * 0.2 + multi_blocks + size_bonus + sparse_linking;

        Ok(DWScore {
            overall_score: overall_score.max(0.0).min(1.0),
            n_linking: linking.len(),
            linking_fraction,
            block_independence: independence,
            avg_block_size,
            linking_density,
            block_sizes,
            linking_constraints: linking,
        })
    }

    /// Identify linking constraints: constraints that reference variables from multiple blocks.
    fn identify_linking_constraints(
        &self,
        matrix: &SparseMatrix,
        partition: &[Vec<usize>],
        row_to_block: &[Option<usize>],
    ) -> Vec<LinkingConstraint> {
        // Build col -> blocks mapping
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

        let mut linking = Vec::new();

        for row in 0..matrix.n_rows {
            // Check if this row's variables span multiple blocks
            let mut blocks_spanned: HashSet<usize> = HashSet::new();
            for &col in &matrix.row_to_cols[row] {
                blocks_spanned.extend(&col_blocks[col]);
            }

            let is_linking = match row_to_block[row] {
                None => true, // not assigned to any block
                Some(_) => blocks_spanned.len() > 1,
            };

            if is_linking && blocks_spanned.len() > 1 {
                let n_vars = matrix.row_to_cols[row].len();
                let density = if matrix.n_cols > 0 {
                    n_vars as f64 / matrix.n_cols as f64
                } else {
                    0.0
                };

                linking.push(LinkingConstraint {
                    index: row,
                    n_blocks_spanned: blocks_spanned.len(),
                    n_variables: n_vars,
                    density,
                });
            }
        }

        // Sort by number of blocks spanned descending
        linking.sort_by(|a, b| b.n_blocks_spanned.cmp(&a.n_blocks_spanned));

        linking
    }

    /// Compute independence between blocks (excluding linking constraints).
    fn compute_block_independence(
        &self,
        matrix: &SparseMatrix,
        partition: &[Vec<usize>],
        linking: &[LinkingConstraint],
    ) -> f64 {
        if partition.len() <= 1 {
            return 0.0;
        }

        let linking_set: HashSet<usize> = linking.iter().map(|l| l.index).collect();

        // Compute variables used by each block (excluding linking constraint vars)
        let mut block_vars: Vec<HashSet<usize>> = Vec::new();
        for block in partition {
            let mut vars = HashSet::new();
            for &row in block {
                if row < matrix.n_rows && !linking_set.contains(&row) {
                    vars.extend(&matrix.row_to_cols[row]);
                }
            }
            block_vars.push(vars);
        }

        let mut independent_pairs = 0usize;
        let mut total_pairs = 0usize;

        for i in 0..block_vars.len() {
            for j in (i + 1)..block_vars.len() {
                let overlap = block_vars[i].intersection(&block_vars[j]).count();
                if overlap == 0 {
                    independent_pairs += 1;
                }
                total_pairs += 1;
            }
        }

        if total_pairs > 0 {
            independent_pairs as f64 / total_pairs as f64
        } else {
            0.0
        }
    }

    /// Compute average density of linking constraints.
    fn compute_linking_density(
        &self,
        _matrix: &SparseMatrix,
        linking: &[LinkingConstraint],
    ) -> f64 {
        if linking.is_empty() {
            return 0.0;
        }
        linking.iter().map(|l| l.density).sum::<f64>() / linking.len() as f64
    }

    /// Score a specific constraint as a potential linking constraint.
    pub fn score_constraint_as_linking(
        &self,
        matrix: &SparseMatrix,
        row: usize,
        partition: &[Vec<usize>],
    ) -> f64 {
        if row >= matrix.n_rows {
            return 0.0;
        }

        // Build col -> blocks mapping for this constraint's variables
        let mut col_blocks: Vec<HashSet<usize>> = vec![HashSet::new(); matrix.n_cols];
        for (block_idx, block) in partition.iter().enumerate() {
            for &r in block {
                if r < matrix.n_rows {
                    for &col in &matrix.row_to_cols[r] {
                        col_blocks[col].insert(block_idx);
                    }
                }
            }
        }

        let mut blocks_touched: HashSet<usize> = HashSet::new();
        for &col in &matrix.row_to_cols[row] {
            blocks_touched.extend(&col_blocks[col]);
        }

        let n_blocks = partition.len() as f64;
        if n_blocks <= 1.0 {
            return 0.0;
        }

        // Score: fraction of blocks this constraint touches
        let touch_fraction = blocks_touched.len() as f64 / n_blocks;

        // Penalize very dense constraints
        let density_penalty = if matrix.n_cols > 0 {
            1.0 - (matrix.row_to_cols[row].len() as f64 / matrix.n_cols as f64).min(1.0) * 0.5
        } else {
            1.0
        };

        touch_fraction * density_penalty
    }

    /// Suggest which constraints should be linking constraints for DW decomposition.
    pub fn suggest_linking_constraints(
        &self,
        matrix: &SparseMatrix,
        partition: &[Vec<usize>],
        max_linking: usize,
    ) -> Vec<usize> {
        let mut candidates: Vec<(usize, f64)> = (0..matrix.n_rows)
            .map(|r| (r, self.score_constraint_as_linking(matrix, r, partition)))
            .filter(|&(_, score)| score > 0.0)
            .collect();

        candidates.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        candidates
            .into_iter()
            .take(max_linking)
            .map(|(idx, _)| idx)
            .collect()
    }

    /// Compute a measure of how well the partition separates the problem.
    pub fn partition_quality(
        &self,
        matrix: &SparseMatrix,
        partition: &[Vec<usize>],
    ) -> f64 {
        if partition.len() <= 1 {
            return 0.0;
        }

        // Balance: all blocks roughly the same size
        let sizes: Vec<f64> = partition.iter().map(|b| b.len() as f64).collect();
        let mean_size = sizes.iter().sum::<f64>() / sizes.len() as f64;
        let cv = if mean_size > 0.0 {
            let var = sizes.iter().map(|&s| (s - mean_size).powi(2)).sum::<f64>() / sizes.len() as f64;
            var.sqrt() / mean_size
        } else {
            1.0
        };
        let balance_score = (1.0 - cv).max(0.0);

        // Coverage: fraction of rows assigned to blocks
        let assigned: usize = partition.iter().map(|b| b.len()).sum();
        let coverage = assigned as f64 / matrix.n_rows.max(1) as f64;

        balance_score * 0.5 + coverage * 0.5
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_dw_matrix() -> (SparseMatrix, Vec<Vec<usize>>) {
        // Two blocks with a linking constraint (row 0) spanning both
        let mut m = SparseMatrix::new(7, 6);
        // Linking constraint: row 0 uses vars from both blocks
        m.add_entry(0, 0);
        m.add_entry(0, 3);
        // Block 1: rows 1-3, vars 0-2
        m.add_entry(1, 0);
        m.add_entry(1, 1);
        m.add_entry(2, 1);
        m.add_entry(2, 2);
        m.add_entry(3, 0);
        m.add_entry(3, 2);
        // Block 2: rows 4-6, vars 3-5
        m.add_entry(4, 3);
        m.add_entry(4, 4);
        m.add_entry(5, 4);
        m.add_entry(5, 5);
        m.add_entry(6, 3);
        m.add_entry(6, 5);

        let partition = vec![vec![1, 2, 3], vec![4, 5, 6]];
        (m, partition)
    }

    #[test]
    fn test_dw_analyze() {
        let (m, partition) = make_dw_matrix();
        let detector = DWDetector::new();
        let score = detector.analyze(&m, &partition).unwrap();

        assert!(score.n_linking >= 1);
        assert!(score.overall_score > 0.0);
    }

    #[test]
    fn test_dw_linking_constraints() {
        let (m, partition) = make_dw_matrix();
        let detector = DWDetector::new();
        let score = detector.analyze(&m, &partition).unwrap();

        // Row 0 should be identified as linking
        assert!(score
            .linking_constraints
            .iter()
            .any(|l| l.index == 0));
    }

    #[test]
    fn test_dw_amenable() {
        let (m, partition) = make_dw_matrix();
        let detector = DWDetector::new();
        let score = detector.analyze(&m, &partition).unwrap();
        assert!(score.is_amenable());
    }

    #[test]
    fn test_dw_summary() {
        let (m, partition) = make_dw_matrix();
        let detector = DWDetector::new();
        let score = detector.analyze(&m, &partition).unwrap();
        let summary = score.summary();
        assert!(summary.contains("DW"));
    }

    #[test]
    fn test_dw_empty_partition() {
        let m = SparseMatrix::new(3, 3);
        let detector = DWDetector::new();
        assert!(detector.analyze(&m, &[]).is_err());
    }

    #[test]
    fn test_dw_no_linking() {
        let mut m = SparseMatrix::new(4, 4);
        m.add_entry(0, 0);
        m.add_entry(1, 1);
        m.add_entry(2, 2);
        m.add_entry(3, 3);
        let partition = vec![vec![0, 1], vec![2, 3]];
        let detector = DWDetector::new();
        let score = detector.analyze(&m, &partition).unwrap();
        assert_eq!(score.n_linking, 0);
    }

    #[test]
    fn test_suggest_linking() {
        let (m, partition) = make_dw_matrix();
        let detector = DWDetector::new();
        let suggested = detector.suggest_linking_constraints(&m, &partition, 5);
        assert!(!suggested.is_empty());
    }

    #[test]
    fn test_partition_quality() {
        let (m, partition) = make_dw_matrix();
        let detector = DWDetector::new();
        let quality = detector.partition_quality(&m, &partition);
        assert!(quality > 0.0 && quality <= 1.0);
    }

    #[test]
    fn test_dw_block_independence() {
        let (m, partition) = make_dw_matrix();
        let detector = DWDetector::new();
        let score = detector.analyze(&m, &partition).unwrap();
        // With linking constraint removed, blocks should be mostly independent
        assert!(score.block_independence > 0.0);
    }

    #[test]
    fn test_score_constraint_as_linking() {
        let (m, partition) = make_dw_matrix();
        let detector = DWDetector::new();
        let score = detector.score_constraint_as_linking(&m, 0, &partition);
        assert!(score > 0.0);
    }

    #[test]
    fn test_linking_constraint_density() {
        let (m, partition) = make_dw_matrix();
        let detector = DWDetector::new();
        let score = detector.analyze(&m, &partition).unwrap();
        assert!(score.linking_density >= 0.0);
    }
}
