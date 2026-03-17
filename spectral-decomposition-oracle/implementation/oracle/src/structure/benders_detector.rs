// Benders amenability detection: identify complicating variables and score
// partitions for Benders decomposition suitability.

use crate::error::{OracleError, OracleResult};
use crate::structure::detector::SparseMatrix;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// A variable identified as complicating for Benders decomposition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplicatingVariable {
    pub index: usize,
    pub coupling_score: f64,
    pub n_blocks_involved: usize,
    pub n_constraints: usize,
}

/// Score for Benders decomposition amenability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BendersScore {
    pub overall_score: f64,
    pub n_complicating: usize,
    pub complicating_fraction: f64,
    pub subproblem_independence: f64,
    pub block_sizes: Vec<usize>,
    pub complicating_variables: Vec<ComplicatingVariable>,
}

impl BendersScore {
    pub fn is_amenable(&self) -> bool {
        self.overall_score > 0.5
    }

    pub fn summary(&self) -> String {
        format!(
            "Benders score: {:.3} | complicating vars: {} ({:.1}%) | independence: {:.3}",
            self.overall_score,
            self.n_complicating,
            self.complicating_fraction * 100.0,
            self.subproblem_independence
        )
    }
}

/// Detector for Benders decomposition amenability.
#[derive(Debug)]
pub struct BendersDetector {
    pub max_complicating_fraction: f64,
    pub min_block_size: usize,
}

impl BendersDetector {
    pub fn new() -> Self {
        Self {
            max_complicating_fraction: 0.2,
            min_block_size: 2,
        }
    }

    /// Analyze a partition for Benders amenability.
    pub fn analyze(
        &self,
        matrix: &SparseMatrix,
        partition: &[Vec<usize>],
    ) -> OracleResult<BendersScore> {
        if partition.is_empty() {
            return Err(OracleError::invalid_input("empty partition"));
        }

        // Compute variable coupling scores
        let coupling_scores = self.compute_coupling_scores(matrix, partition);

        // Identify complicating variables (variables appearing in multiple blocks)
        let complicating = self.identify_complicating_variables(matrix, partition, &coupling_scores);

        let n_total_vars = matrix.n_cols;
        let complicating_fraction = if n_total_vars > 0 {
            complicating.len() as f64 / n_total_vars as f64
        } else {
            1.0
        };

        // Compute subproblem independence when complicating vars are fixed
        let independence = self.compute_subproblem_independence(matrix, partition, &complicating);

        let block_sizes: Vec<usize> = partition.iter().map(|b| b.len()).collect();

        // Overall score
        let few_complicating = (1.0 - complicating_fraction / self.max_complicating_fraction)
            .max(0.0)
            .min(1.0);
        let multi_blocks = if partition.len() > 1 { 0.2 } else { 0.0 };
        let size_bonus = if block_sizes.iter().all(|&s| s >= self.min_block_size) {
            0.1
        } else {
            0.0
        };

        let overall_score = few_complicating * 0.4 + independence * 0.3 + multi_blocks + size_bonus;

        Ok(BendersScore {
            overall_score: overall_score.max(0.0).min(1.0),
            n_complicating: complicating.len(),
            complicating_fraction,
            subproblem_independence: independence,
            block_sizes,
            complicating_variables: complicating,
        })
    }

    /// Compute coupling score for each variable: how many blocks it touches.
    fn compute_coupling_scores(
        &self,
        matrix: &SparseMatrix,
        partition: &[Vec<usize>],
    ) -> Vec<f64> {
        let n_cols = matrix.n_cols;
        let n_blocks = partition.len() as f64;
        let mut scores = vec![0.0_f64; n_cols];

        // Build col -> blocks mapping
        let col_blocks = self.column_block_participation(matrix, partition);

        for col in 0..n_cols {
            let blocks_touched = col_blocks[col].len() as f64;
            // Coupling score: fraction of blocks this variable touches
            scores[col] = if n_blocks > 1.0 {
                (blocks_touched - 1.0) / (n_blocks - 1.0)
            } else {
                0.0
            };
        }

        scores
    }

    /// Build mapping: column index -> set of block indices it participates in.
    fn column_block_participation(
        &self,
        matrix: &SparseMatrix,
        partition: &[Vec<usize>],
    ) -> Vec<HashSet<usize>> {
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

        col_blocks
    }

    /// Identify complicating variables: those appearing in >1 block.
    fn identify_complicating_variables(
        &self,
        matrix: &SparseMatrix,
        partition: &[Vec<usize>],
        coupling_scores: &[f64],
    ) -> Vec<ComplicatingVariable> {
        let col_blocks = self.column_block_participation(matrix, partition);

        let mut complicating = Vec::new();

        for col in 0..matrix.n_cols {
            let n_blocks = col_blocks[col].len();
            if n_blocks > 1 {
                complicating.push(ComplicatingVariable {
                    index: col,
                    coupling_score: coupling_scores[col],
                    n_blocks_involved: n_blocks,
                    n_constraints: matrix.col_to_rows[col].len(),
                });
            }
        }

        // Sort by coupling score descending
        complicating.sort_by(|a, b| {
            b.coupling_score
                .partial_cmp(&a.coupling_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        complicating
    }

    /// Compute how independent subproblems become when complicating variables are fixed.
    fn compute_subproblem_independence(
        &self,
        matrix: &SparseMatrix,
        partition: &[Vec<usize>],
        complicating: &[ComplicatingVariable],
    ) -> f64 {
        if partition.len() <= 1 {
            return 0.0;
        }

        let comp_set: HashSet<usize> = complicating.iter().map(|c| c.index).collect();

        // For each pair of blocks, check if they share non-complicating variables
        let mut block_vars: Vec<HashSet<usize>> = Vec::new();
        for block in partition {
            let mut vars = HashSet::new();
            for &row in block {
                if row < matrix.n_rows {
                    for &col in &matrix.row_to_cols[row] {
                        if !comp_set.contains(&col) {
                            vars.insert(col);
                        }
                    }
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

    /// Rank variables by their suitability as Benders complicating variables.
    pub fn rank_complicating_candidates(
        &self,
        matrix: &SparseMatrix,
        partition: &[Vec<usize>],
    ) -> Vec<ComplicatingVariable> {
        let coupling_scores = self.compute_coupling_scores(matrix, partition);
        let mut candidates = self.identify_complicating_variables(matrix, partition, &coupling_scores);

        // Also include highly-connected single-block variables as potential candidates
        for col in 0..matrix.n_cols {
            if coupling_scores[col] == 0.0 && matrix.col_to_rows[col].len() > 3 {
                candidates.push(ComplicatingVariable {
                    index: col,
                    coupling_score: 0.0,
                    n_blocks_involved: 1,
                    n_constraints: matrix.col_to_rows[col].len(),
                });
            }
        }

        // Sort by coupling score then by constraint count
        candidates.sort_by(|a, b| {
            b.coupling_score
                .partial_cmp(&a.coupling_score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.n_constraints.cmp(&a.n_constraints))
        });

        candidates
    }

    /// Suggest an improved partition for Benders decomposition.
    pub fn suggest_partition(
        &self,
        matrix: &SparseMatrix,
        initial_partition: &[Vec<usize>],
        max_complicating: usize,
    ) -> Vec<Vec<usize>> {
        // Start with initial partition, then try to reduce complicating variables
        let coupling_scores = self.compute_coupling_scores(matrix, initial_partition);
        let complicating = self.identify_complicating_variables(matrix, initial_partition, &coupling_scores);

        if complicating.len() <= max_complicating {
            return initial_partition.to_vec();
        }

        // Merge blocks that share many complicating variables
        let merged = initial_partition.to_vec();
        let col_blocks = self.column_block_participation(matrix, initial_partition);

        // Find the pair of blocks with most shared complicating variables
        let n_blocks = merged.len();
        if n_blocks < 2 {
            return merged;
        }

        let mut best_merge = (0, 1);
        let mut best_shared = 0usize;

        for i in 0..n_blocks {
            for j in (i + 1)..n_blocks {
                let mut shared = 0;
                for cv in &complicating {
                    if col_blocks[cv.index].contains(&i) && col_blocks[cv.index].contains(&j) {
                        shared += 1;
                    }
                }
                if shared > best_shared {
                    best_shared = shared;
                    best_merge = (i, j);
                }
            }
        }

        // Merge the two blocks
        let (i, j) = best_merge;
        let merge_block: Vec<usize> = merged[i]
            .iter()
            .chain(merged[j].iter())
            .copied()
            .collect();

        let mut new_partition = Vec::new();
        for (idx, block) in merged.iter().enumerate() {
            if idx == i {
                new_partition.push(merge_block.clone());
            } else if idx != j {
                new_partition.push(block.clone());
            }
        }

        new_partition
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_benders_matrix() -> (SparseMatrix, Vec<Vec<usize>>) {
        // Two blocks with coupling variable (col 4)
        let mut m = SparseMatrix::new(6, 5);
        // Block 1: rows 0-2, vars 0-1, coupling var 4
        m.add_entry(0, 0);
        m.add_entry(0, 4);
        m.add_entry(1, 1);
        m.add_entry(2, 0);
        m.add_entry(2, 1);
        // Block 2: rows 3-5, vars 2-3, coupling var 4
        m.add_entry(3, 2);
        m.add_entry(3, 4);
        m.add_entry(4, 3);
        m.add_entry(5, 2);
        m.add_entry(5, 3);

        let partition = vec![vec![0, 1, 2], vec![3, 4, 5]];
        (m, partition)
    }

    #[test]
    fn test_benders_analyze() {
        let (m, partition) = make_benders_matrix();
        let detector = BendersDetector::new();
        let score = detector.analyze(&m, &partition).unwrap();

        assert!(score.n_complicating >= 1);
        assert!(score.complicating_fraction > 0.0);
        assert!(score.overall_score > 0.0);
    }

    #[test]
    fn test_benders_complicating_vars() {
        let (m, partition) = make_benders_matrix();
        let detector = BendersDetector::new();
        let score = detector.analyze(&m, &partition).unwrap();

        assert!(score.complicating_variables.iter().any(|v| v.index == 4));
    }

    #[test]
    fn test_benders_independence() {
        let (m, partition) = make_benders_matrix();
        let detector = BendersDetector::new();
        let score = detector.analyze(&m, &partition).unwrap();

        // After removing complicating var 4, blocks should be independent
        assert!(score.subproblem_independence > 0.5);
    }

    #[test]
    fn test_benders_amenable() {
        let (m, partition) = make_benders_matrix();
        let detector = BendersDetector::new();
        let score = detector.analyze(&m, &partition).unwrap();
        // With only 1 complicating variable out of 5, should be amenable
        assert!(score.is_amenable());
    }

    #[test]
    fn test_benders_summary() {
        let (m, partition) = make_benders_matrix();
        let detector = BendersDetector::new();
        let score = detector.analyze(&m, &partition).unwrap();
        let summary = score.summary();
        assert!(summary.contains("Benders"));
    }

    #[test]
    fn test_benders_empty_partition() {
        let m = SparseMatrix::new(3, 3);
        let detector = BendersDetector::new();
        assert!(detector.analyze(&m, &[]).is_err());
    }

    #[test]
    fn test_rank_complicating_candidates() {
        let (m, partition) = make_benders_matrix();
        let detector = BendersDetector::new();
        let candidates = detector.rank_complicating_candidates(&m, &partition);
        assert!(!candidates.is_empty());
        // Most coupling variable should be first
        assert_eq!(candidates[0].index, 4);
    }

    #[test]
    fn test_suggest_partition() {
        let (m, partition) = make_benders_matrix();
        let detector = BendersDetector::new();
        let suggested = detector.suggest_partition(&m, &partition, 0);
        // Should have merged some blocks
        assert!(suggested.len() <= partition.len());
    }

    #[test]
    fn test_benders_no_coupling() {
        let mut m = SparseMatrix::new(4, 4);
        m.add_entry(0, 0);
        m.add_entry(1, 1);
        m.add_entry(2, 2);
        m.add_entry(3, 3);
        let partition = vec![vec![0, 1], vec![2, 3]];
        let detector = BendersDetector::new();
        let score = detector.analyze(&m, &partition).unwrap();
        assert_eq!(score.n_complicating, 0);
    }

    #[test]
    fn test_coupling_scores() {
        let (m, partition) = make_benders_matrix();
        let detector = BendersDetector::new();
        let scores = detector.compute_coupling_scores(&m, &partition);
        assert_eq!(scores.len(), 5);
        assert!(scores[4] > 0.0); // coupling variable
        assert_eq!(scores[0], 0.0); // block-local variable
    }

    #[test]
    fn test_column_block_participation() {
        let (m, partition) = make_benders_matrix();
        let detector = BendersDetector::new();
        let col_blocks = detector.column_block_participation(&m, &partition);
        assert_eq!(col_blocks[4].len(), 2); // appears in both blocks
        assert_eq!(col_blocks[0].len(), 1); // only in block 0
    }
}
