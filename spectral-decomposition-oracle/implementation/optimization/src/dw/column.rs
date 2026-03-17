// ---------------------------------------------------------------------------
// Column management for Dantzig-Wolfe decomposition
// ---------------------------------------------------------------------------

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use indexmap::IndexSet;
use log::{debug, warn};

use crate::error::{OptError, OptResult};
use crate::lp::LpProblem;

// ---------------------------------------------------------------------------
// DWColumn
// ---------------------------------------------------------------------------

/// A column (extreme point) in the Dantzig-Wolfe restricted master problem.
///
/// Each column represents a point `x_k` from the polyhedron of a specific
/// subproblem block.  The master problem uses convex combination weights
/// (lambda variables) over these columns.
#[derive(Debug, Clone)]
pub struct DWColumn {
    /// Block (subproblem) index this column belongs to.
    pub block: usize,
    /// Extreme point of the subproblem polyhedron.
    pub point: Vec<f64>,
    /// Original objective cost:  c_k^T * x_k.
    pub original_cost: f64,
    /// Coefficients in the linking constraints:  A_k * x_k.
    pub linking_coefficients: Vec<f64>,
    /// Number of iterations since this column was last in the basis.
    pub age: usize,
    /// Number of times this column has been in the RMP basis.
    pub times_in_basis: usize,
    /// Reduced cost at the iteration this column was generated.
    pub reduced_cost_at_generation: f64,
    /// Unique identifier.
    pub id: usize,
}

impl DWColumn {
    /// Utility score: lower (more negative) reduced cost and higher basis
    /// participation are more useful.  Used for retention during cleanup.
    pub fn utility_score(&self) -> f64 {
        // Combine recency (inverse age) and basis participation.
        let age_penalty = self.age as f64;
        let basis_bonus = self.times_in_basis as f64 * 10.0;
        basis_bonus - age_penalty + (-self.reduced_cost_at_generation).max(0.0)
    }
}

// ---------------------------------------------------------------------------
// ColumnPool
// ---------------------------------------------------------------------------

/// Pool of columns generated during column generation.
///
/// Provides de-duplication via hashing, per-block indexing, and ageing /
/// cleanup utilities.
pub struct ColumnPool {
    columns: Vec<DWColumn>,
    column_hash: IndexSet<u64>,
    next_id: usize,
    capacity: usize,
}

impl std::fmt::Debug for ColumnPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ColumnPool")
            .field("num_columns", &self.columns.len())
            .field("capacity", &self.capacity)
            .field("next_id", &self.next_id)
            .finish()
    }
}

impl ColumnPool {
    /// Create a new column pool with the given maximum capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            columns: Vec::with_capacity(capacity.min(1024)),
            column_hash: IndexSet::with_capacity(capacity.min(1024)),
            next_id: 0,
            capacity,
        }
    }

    /// Add a column to the pool if it is not a duplicate.
    ///
    /// Returns `Some(index)` if the column was added, `None` if it was a
    /// duplicate or the pool is at capacity.
    pub fn add(&mut self, mut column: DWColumn) -> Option<usize> {
        let hash = Self::hash_column(&column);
        if self.column_hash.contains(&hash) {
            debug!("Duplicate column detected (hash={}), skipping", hash);
            return None;
        }
        if self.columns.len() >= self.capacity {
            warn!(
                "Column pool at capacity ({}), cannot add more",
                self.capacity
            );
            return None;
        }
        column.id = self.next_id;
        self.next_id += 1;
        let idx = self.columns.len();
        self.column_hash.insert(hash);
        self.columns.push(column);
        Some(idx)
    }

    /// Hash a column based on its block index and discretised point values.
    ///
    /// We discretise to 8 decimal places to avoid floating-point noise
    /// causing spurious duplicates.
    pub fn hash_column(column: &DWColumn) -> u64 {
        let mut hasher = DefaultHasher::new();
        column.block.hash(&mut hasher);
        for &v in &column.point {
            // Discretise: round to 8 decimal places, convert to integer bits.
            let discretised = (v * 1e8).round() as i64;
            discretised.hash(&mut hasher);
        }
        hasher.finish()
    }

    /// Get a reference to the column at the given index.
    pub fn get(&self, index: usize) -> Option<&DWColumn> {
        self.columns.get(index)
    }

    /// Get a mutable reference to the column at the given index.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut DWColumn> {
        self.columns.get_mut(index)
    }

    /// Number of columns in the pool.
    pub fn len(&self) -> usize {
        self.columns.len()
    }

    /// Whether the pool is empty.
    pub fn is_empty(&self) -> bool {
        self.columns.is_empty()
    }

    /// Return the indices of all columns belonging to the given block.
    pub fn columns_for_block(&self, block: usize) -> Vec<usize> {
        self.columns
            .iter()
            .enumerate()
            .filter(|(_, c)| c.block == block)
            .map(|(i, _)| i)
            .collect()
    }

    /// Remove columns older than `age_limit` that have been in the basis
    /// fewer than `min_basis_count` times.
    pub fn cleanup(&mut self, age_limit: usize, min_basis_count: usize) {
        let before = self.columns.len();
        // Rebuild hash set and column list, keeping only retained columns.
        let mut retained_columns = Vec::with_capacity(self.columns.len());
        let mut retained_hashes = IndexSet::with_capacity(self.columns.len());

        for col in self.columns.drain(..) {
            if col.age <= age_limit || col.times_in_basis >= min_basis_count {
                let hash = Self::hash_column(&col);
                retained_hashes.insert(hash);
                retained_columns.push(col);
            }
        }

        self.columns = retained_columns;
        self.column_hash = retained_hashes;
        let removed = before - self.columns.len();
        if removed > 0 {
            debug!(
                "Column cleanup: removed {} columns, {} remain",
                removed,
                self.columns.len()
            );
        }
    }

    /// Increment the age of every column in the pool.
    pub fn age_all(&mut self) {
        for col in &mut self.columns {
            col.age += 1;
        }
    }

    /// Mark the columns at the given indices as being in the current basis.
    /// Resets their age to 0 and increments their basis count.
    pub fn mark_in_basis(&mut self, indices: &[usize]) {
        for &idx in indices {
            if let Some(col) = self.columns.get_mut(idx) {
                col.age = 0;
                col.times_in_basis += 1;
            }
        }
    }

    /// Return the index of the column with the lowest original cost for
    /// the given block, or `None` if the block has no columns.
    pub fn best_column_for_block(&self, block: usize) -> Option<usize> {
        self.columns
            .iter()
            .enumerate()
            .filter(|(_, c)| c.block == block)
            .min_by(|(_, a), (_, b)| {
                a.original_cost
                    .partial_cmp(&b.original_cost)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
    }

    /// Create an artificial / Farkas column for Phase I of the DW master.
    pub fn farkas_column(
        block: usize,
        point: Vec<f64>,
        linking_coeffs: Vec<f64>,
    ) -> DWColumn {
        DWColumn {
            block,
            original_cost: 1e6, // large penalty cost
            linking_coefficients: linking_coeffs,
            age: 0,
            times_in_basis: 0,
            reduced_cost_at_generation: 0.0,
            id: 0, // will be assigned by pool
            point,
        }
    }

    /// Generate initial columns for each block from simple heuristics.
    ///
    /// For each block we generate columns from:
    ///   1. The zero point (if feasible for the subproblem bounds).
    ///   2. The midpoint of variable bounds (clamped to finite values).
    ///   3. Lower-bound corner point.
    ///
    /// The `partition` slice assigns each variable index to a block index
    /// (values in `0..num_blocks`).
    pub fn initial_columns_from_lp(
        problem: &LpProblem,
        partition: &[usize],
        num_blocks: usize,
    ) -> OptResult<Vec<DWColumn>> {
        if partition.len() != problem.num_vars {
            return Err(OptError::InvalidProblem {
                reason: format!(
                    "partition length {} != num_vars {}",
                    partition.len(),
                    problem.num_vars
                ),
            });
        }

        // Identify which variables belong to each block.
        let mut block_vars: Vec<Vec<usize>> = vec![Vec::new(); num_blocks];
        for (j, &b) in partition.iter().enumerate() {
            if b < num_blocks {
                block_vars[b].push(j);
            }
        }

        // Identify linking constraints: constraints that reference variables
        // from more than one block.
        let mut linking_rows: Vec<bool> = vec![false; problem.num_constraints];
        for i in 0..problem.num_constraints {
            let rs = problem.row_starts[i];
            let re = problem.row_starts[i + 1];
            let mut seen_block: Option<usize> = None;
            let mut is_linking = false;
            for k in rs..re {
                let var = problem.col_indices[k];
                if var < partition.len() {
                    let b = partition[var];
                    match seen_block {
                        None => seen_block = Some(b),
                        Some(prev) if prev != b => {
                            is_linking = true;
                            break;
                        }
                        _ => {}
                    }
                }
            }
            linking_rows[i] = is_linking;
        }
        let linking_indices: Vec<usize> = linking_rows
            .iter()
            .enumerate()
            .filter(|(_, &is_link)| is_link)
            .map(|(i, _)| i)
            .collect();
        let _num_linking = linking_indices.len();

        let mut columns = Vec::new();

        for block in 0..num_blocks {
            let vars = &block_vars[block];
            if vars.is_empty() {
                continue;
            }

            // Strategy 1: lower-bound point
            let mut lb_point = vec![0.0; problem.num_vars];
            for &j in vars {
                lb_point[j] = if problem.lower_bounds[j].is_finite() {
                    problem.lower_bounds[j]
                } else {
                    0.0
                };
            }
            let lb_cost = compute_original_cost(problem, &lb_point);
            let lb_linking = compute_linking_coefficients(problem, &lb_point, &linking_indices);
            columns.push(DWColumn {
                block,
                point: lb_point.clone(),
                original_cost: lb_cost,
                linking_coefficients: lb_linking,
                age: 0,
                times_in_basis: 0,
                reduced_cost_at_generation: 0.0,
                id: 0,
            });

            // Strategy 2: midpoint of bounds (clamped)
            let mut mid_point = vec![0.0; problem.num_vars];
            for &j in vars {
                let lo = if problem.lower_bounds[j].is_finite() {
                    problem.lower_bounds[j]
                } else {
                    -100.0
                };
                let hi = if problem.upper_bounds[j].is_finite() {
                    problem.upper_bounds[j]
                } else {
                    100.0
                };
                mid_point[j] = (lo + hi) / 2.0;
            }
            let mid_cost = compute_original_cost(problem, &mid_point);
            let mid_linking =
                compute_linking_coefficients(problem, &mid_point, &linking_indices);
            columns.push(DWColumn {
                block,
                point: mid_point,
                original_cost: mid_cost,
                linking_coefficients: mid_linking,
                age: 0,
                times_in_basis: 0,
                reduced_cost_at_generation: 0.0,
                id: 0,
            });
        }

        debug!(
            "Generated {} initial columns for {} blocks",
            columns.len(),
            num_blocks
        );
        Ok(columns)
    }

    /// Keep only the most useful columns per block, discarding the rest.
    ///
    /// Retains at most `max_columns_per_block` columns for each block,
    /// ranked by `utility_score()`.
    pub fn compress(&mut self, max_columns_per_block: usize) {
        // Determine how many blocks we have.
        let max_block = self.columns.iter().map(|c| c.block).max().unwrap_or(0);

        let mut retained = Vec::new();
        let mut retained_hashes = IndexSet::new();

        for block in 0..=max_block {
            let mut block_cols: Vec<(usize, f64)> = self
                .columns
                .iter()
                .enumerate()
                .filter(|(_, c)| c.block == block)
                .map(|(i, c)| (i, c.utility_score()))
                .collect();

            // Sort descending by utility score.
            block_cols.sort_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });

            for (idx, _) in block_cols.into_iter().take(max_columns_per_block) {
                let col = self.columns[idx].clone();
                let hash = Self::hash_column(&col);
                retained_hashes.insert(hash);
                retained.push(col);
            }
        }

        let before = self.columns.len();
        self.columns = retained;
        self.column_hash = retained_hashes;
        let removed = before - self.columns.len();
        if removed > 0 {
            debug!(
                "Column compress: removed {} columns, {} remain",
                removed,
                self.columns.len()
            );
        }
    }

    /// Immutable slice of all columns.
    pub fn all_columns(&self) -> &[DWColumn] {
        &self.columns
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Compute the original objective cost c^T * x.
fn compute_original_cost(problem: &LpProblem, point: &[f64]) -> f64 {
    problem
        .obj_coeffs
        .iter()
        .zip(point.iter())
        .map(|(c, x)| c * x)
        .sum()
}

/// Compute A_linking * x  for the linking constraint rows.
fn compute_linking_coefficients(
    problem: &LpProblem,
    point: &[f64],
    linking_indices: &[usize],
) -> Vec<f64> {
    linking_indices
        .iter()
        .map(|&row| {
            let rs = problem.row_starts[row];
            let re = problem.row_starts[row + 1];
            let mut sum = 0.0;
            for k in rs..re {
                let var = problem.col_indices[k];
                if var < point.len() {
                    sum += problem.values[k] * point[var];
                }
            }
            sum
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lp::{ConstraintType, LpProblem};

    fn make_column(block: usize, point: Vec<f64>, cost: f64, linking: Vec<f64>) -> DWColumn {
        DWColumn {
            block,
            point,
            original_cost: cost,
            linking_coefficients: linking,
            age: 0,
            times_in_basis: 0,
            reduced_cost_at_generation: 0.0,
            id: 0,
        }
    }

    #[test]
    fn test_pool_new() {
        let pool = ColumnPool::new(100);
        assert_eq!(pool.len(), 0);
        assert!(pool.is_empty());
    }

    #[test]
    fn test_add_column() {
        let mut pool = ColumnPool::new(100);
        let col = make_column(0, vec![1.0, 2.0], 3.0, vec![1.0]);
        let idx = pool.add(col);
        assert_eq!(idx, Some(0));
        assert_eq!(pool.len(), 1);
        assert!(!pool.is_empty());
    }

    #[test]
    fn test_duplicate_detection() {
        let mut pool = ColumnPool::new(100);
        let col1 = make_column(0, vec![1.0, 2.0], 3.0, vec![1.0]);
        let col2 = make_column(0, vec![1.0, 2.0], 5.0, vec![2.0]); // same block+point
        assert!(pool.add(col1).is_some());
        assert!(pool.add(col2).is_none()); // duplicate
        assert_eq!(pool.len(), 1);
    }

    #[test]
    fn test_different_points_not_duplicate() {
        let mut pool = ColumnPool::new(100);
        let col1 = make_column(0, vec![1.0, 2.0], 3.0, vec![1.0]);
        let col2 = make_column(0, vec![1.0, 3.0], 4.0, vec![2.0]);
        assert!(pool.add(col1).is_some());
        assert!(pool.add(col2).is_some());
        assert_eq!(pool.len(), 2);
    }

    #[test]
    fn test_capacity_limit() {
        let mut pool = ColumnPool::new(2);
        let c1 = make_column(0, vec![1.0], 1.0, vec![]);
        let c2 = make_column(0, vec![2.0], 2.0, vec![]);
        let c3 = make_column(0, vec![3.0], 3.0, vec![]);
        assert!(pool.add(c1).is_some());
        assert!(pool.add(c2).is_some());
        assert!(pool.add(c3).is_none()); // at capacity
    }

    #[test]
    fn test_get_column() {
        let mut pool = ColumnPool::new(100);
        let col = make_column(0, vec![1.0, 2.0], 3.0, vec![1.0]);
        pool.add(col);
        let c = pool.get(0).unwrap();
        assert_eq!(c.block, 0);
        assert!((c.original_cost - 3.0).abs() < 1e-12);
        assert!(pool.get(1).is_none());
    }

    #[test]
    fn test_columns_for_block() {
        let mut pool = ColumnPool::new(100);
        pool.add(make_column(0, vec![1.0], 1.0, vec![]));
        pool.add(make_column(1, vec![2.0], 2.0, vec![]));
        pool.add(make_column(0, vec![3.0], 3.0, vec![]));
        pool.add(make_column(1, vec![4.0], 4.0, vec![]));
        pool.add(make_column(2, vec![5.0], 5.0, vec![]));

        assert_eq!(pool.columns_for_block(0), vec![0, 2]);
        assert_eq!(pool.columns_for_block(1), vec![1, 3]);
        assert_eq!(pool.columns_for_block(2), vec![4]);
        assert!(pool.columns_for_block(3).is_empty());
    }

    #[test]
    fn test_age_all_and_mark_in_basis() {
        let mut pool = ColumnPool::new(100);
        pool.add(make_column(0, vec![1.0], 1.0, vec![]));
        pool.add(make_column(0, vec![2.0], 2.0, vec![]));

        pool.age_all();
        assert_eq!(pool.get(0).unwrap().age, 1);
        assert_eq!(pool.get(1).unwrap().age, 1);

        pool.age_all();
        assert_eq!(pool.get(0).unwrap().age, 2);

        pool.mark_in_basis(&[0]);
        assert_eq!(pool.get(0).unwrap().age, 0);
        assert_eq!(pool.get(0).unwrap().times_in_basis, 1);
        assert_eq!(pool.get(1).unwrap().age, 2); // unchanged
    }

    #[test]
    fn test_cleanup() {
        let mut pool = ColumnPool::new(100);
        pool.add(make_column(0, vec![1.0], 1.0, vec![]));
        pool.add(make_column(0, vec![2.0], 2.0, vec![]));
        pool.add(make_column(0, vec![3.0], 3.0, vec![]));

        // Age columns, but mark the first in basis.
        for _ in 0..5 {
            pool.age_all();
        }
        pool.mark_in_basis(&[0]);
        // Now: col 0 has age=0, basis=1; cols 1,2 have age=5, basis=0.

        pool.cleanup(3, 1); // age_limit=3, min_basis_count=1
        // Cols 1 and 2 should be removed (age 5 > 3, basis 0 < 1).
        assert_eq!(pool.len(), 1);
        assert_eq!(pool.get(0).unwrap().times_in_basis, 1);
    }

    #[test]
    fn test_best_column_for_block() {
        let mut pool = ColumnPool::new(100);
        pool.add(make_column(0, vec![1.0], 5.0, vec![]));
        pool.add(make_column(0, vec![2.0], 2.0, vec![])); // cheapest
        pool.add(make_column(0, vec![3.0], 8.0, vec![]));
        pool.add(make_column(1, vec![4.0], 1.0, vec![]));

        assert_eq!(pool.best_column_for_block(0), Some(1));
        assert_eq!(pool.best_column_for_block(1), Some(3));
        assert_eq!(pool.best_column_for_block(2), None);
    }

    #[test]
    fn test_farkas_column() {
        let col = ColumnPool::farkas_column(2, vec![0.0, 0.0], vec![1.0, 2.0]);
        assert_eq!(col.block, 2);
        assert!((col.original_cost - 1e6).abs() < 1e-6);
        assert_eq!(col.linking_coefficients, vec![1.0, 2.0]);
    }

    #[test]
    fn test_compress() {
        let mut pool = ColumnPool::new(100);
        for i in 0..10 {
            let mut col = make_column(0, vec![i as f64], i as f64, vec![]);
            col.reduced_cost_at_generation = -(i as f64); // higher index → more negative → better
            pool.add(col);
        }
        pool.compress(3);
        assert_eq!(pool.len(), 3);
    }

    #[test]
    fn test_hash_column_deterministic() {
        let col1 = make_column(0, vec![1.0, 2.0, 3.0], 1.0, vec![]);
        let col2 = make_column(0, vec![1.0, 2.0, 3.0], 5.0, vec![99.0]);
        // Same block and point → same hash.
        assert_eq!(
            ColumnPool::hash_column(&col1),
            ColumnPool::hash_column(&col2)
        );
    }

    #[test]
    fn test_initial_columns_from_lp() {
        // Build a small LP with 4 variables, 2 blocks, and 1 linking constraint.
        let mut lp = LpProblem::new(false);
        // Block 0: x0, x1
        lp.add_variable(1.0, 0.0, 10.0, None);
        lp.add_variable(2.0, 0.0, 10.0, None);
        // Block 1: x2, x3
        lp.add_variable(3.0, 0.0, 10.0, None);
        lp.add_variable(4.0, 0.0, 10.0, None);
        // Linking: x0 + x2 <= 5
        lp.add_constraint(&[0, 2], &[1.0, 1.0], ConstraintType::Le, 5.0)
            .unwrap();
        // Block 0 local: x0 + x1 <= 8
        lp.add_constraint(&[0, 1], &[1.0, 1.0], ConstraintType::Le, 8.0)
            .unwrap();
        // Block 1 local: x2 + x3 <= 7
        lp.add_constraint(&[2, 3], &[1.0, 1.0], ConstraintType::Le, 7.0)
            .unwrap();

        let partition = vec![0, 0, 1, 1];
        let columns = ColumnPool::initial_columns_from_lp(&lp, &partition, 2).unwrap();
        // 2 strategies × 2 blocks = 4 columns
        assert_eq!(columns.len(), 4);
        // Each column should have 1 linking coefficient (for the linking row).
        for col in &columns {
            assert_eq!(col.linking_coefficients.len(), 1);
        }
    }

    #[test]
    fn test_initial_columns_partition_mismatch() {
        let mut lp = LpProblem::new(false);
        lp.add_variable(1.0, 0.0, 10.0, None);
        let partition = vec![0, 0]; // length 2 != num_vars 1
        let result = ColumnPool::initial_columns_from_lp(&lp, &partition, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_utility_score() {
        let mut col = make_column(0, vec![1.0], 1.0, vec![]);
        col.times_in_basis = 5;
        col.age = 2;
        col.reduced_cost_at_generation = -3.0;
        let score = col.utility_score();
        // 5*10 - 2 + 3.0 = 51.0
        assert!((score - 51.0).abs() < 1e-12);
    }
}
