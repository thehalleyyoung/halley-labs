//! L3 Partition-to-Bound Bridge certificate.
//!
//! Given partition P = {B_1,...,B_k}, computes the L3 bound:
//!   z_LP - z_D(P) ≤ Σ_{e ∈ E_cross(P)} |y*_e| * (n_e(P) - 1)
//!
//! where E_cross(P) is the set of crossing edges (constraints involving variables
//! from multiple blocks), y*_e is the optimal dual value for constraint e, and
//! n_e(P) is the number of blocks that constraint e touches.

use crate::error::{CertificateError, CertificateResult};
use chrono::Utc;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A crossing edge: a constraint that spans variables from multiple partition blocks.
///
/// Its contribution to the L3 bound is `|y*_e| * (n_e(P) − 1)`,
/// where `y*_e` is the optimal dual value and `n_e(P)` is the block count.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossingEdge {
    pub constraint_index: usize,
    pub constraint_name: String,
    pub blocks_touched: Vec<usize>,
    pub num_blocks_touched: usize,
    pub dual_value: f64,
    pub contribution: f64,
}

impl CrossingEdge {
    /// Compute the contribution of this crossing edge to the L3 bound.
    /// Contribution = |y*_e| * (n_e(P) - 1)
    pub fn compute_contribution(&mut self) {
        self.contribution = self.dual_value.abs() * (self.num_blocks_touched as f64 - 1.0);
    }

    /// Whether this edge contributes significantly to the bound.
    pub fn is_significant(&self, threshold: f64) -> bool {
        self.contribution > threshold
    }
}

/// Verification status of a certificate.
///
/// Certificates start as [`Unverified`](Self::Unverified), transition to
/// [`Verified`](Self::Verified) after independent checking, or
/// [`Failed`](Self::Failed) if the bound is violated. A certificate becomes
/// [`Stale`](Self::Stale) when the underlying data has changed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerificationStatus {
    Unverified,
    Verified,
    Failed,
    Stale,
}

impl std::fmt::Display for VerificationStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unverified => write!(f, "UNVERIFIED"),
            Self::Verified => write!(f, "VERIFIED"),
            Self::Failed => write!(f, "FAILED"),
            Self::Stale => write!(f, "STALE"),
        }
    }
}

/// Method used to generate the variable partition.
///
/// Tracks provenance so that certificate consumers can assess whether the
/// partition was produced by a spectral, graph-partitioning, or heuristic method.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PartitionMethod {
    Spectral,
    GCG,
    Manual,
    KMeans,
    Metis,
    Random,
    Custom(String),
}

impl std::fmt::Display for PartitionMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Spectral => write!(f, "spectral"),
            Self::GCG => write!(f, "GCG"),
            Self::Manual => write!(f, "manual"),
            Self::KMeans => write!(f, "k-means"),
            Self::Metis => write!(f, "METIS"),
            Self::Random => write!(f, "random"),
            Self::Custom(name) => write!(f, "custom({})", name),
        }
    }
}

/// Partition of MIP variables into blocks for decomposition.
///
/// Each variable is assigned to exactly one block. The partition tracks block
/// sizes and the method used for construction. Used by the L3 certificate
/// to identify crossing edges.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Partition {
    pub num_variables: usize,
    pub num_blocks: usize,
    pub block_assignments: Vec<usize>,
    pub block_sizes: Vec<usize>,
    pub method: PartitionMethod,
}

impl Partition {
    /// Create a new partition from block assignments.
    pub fn new(
        assignments: Vec<usize>,
        num_blocks: usize,
        method: PartitionMethod,
    ) -> CertificateResult<Self> {
        let num_variables = assignments.len();
        if num_variables == 0 {
            return Err(CertificateError::invalid_partition("empty assignment vector"));
        }

        let max_block = assignments.iter().cloned().max().unwrap_or(0);
        if max_block >= num_blocks {
            return Err(CertificateError::invalid_partition_with_details(
                format!(
                    "assignment {} exceeds block count {}",
                    max_block, num_blocks
                ),
                num_blocks,
                num_variables,
            ));
        }

        let mut block_sizes = vec![0usize; num_blocks];
        for &b in &assignments {
            block_sizes[b] += 1;
        }

        for (i, &sz) in block_sizes.iter().enumerate() {
            if sz == 0 {
                return Err(CertificateError::invalid_partition(format!(
                    "block {} is empty",
                    i
                )));
            }
        }

        Ok(Self {
            num_variables,
            num_blocks,
            block_assignments: assignments,
            block_sizes,
            method,
        })
    }

    /// Returns the block index for a given variable.
    pub fn block_of(&self, var: usize) -> Option<usize> {
        self.block_assignments.get(var).copied()
    }

    /// Returns all variables in a given block.
    pub fn variables_in_block(&self, block: usize) -> Vec<usize> {
        self.block_assignments
            .iter()
            .enumerate()
            .filter(|(_, &b)| b == block)
            .map(|(i, _)| i)
            .collect()
    }

    /// Balance ratio: min_block_size / max_block_size.
    pub fn balance_ratio(&self) -> f64 {
        let min = *self.block_sizes.iter().min().unwrap_or(&1) as f64;
        let max = *self.block_sizes.iter().max().unwrap_or(&1) as f64;
        if max < 1e-15 {
            return 0.0;
        }
        min / max
    }

    /// Checks how many blocks a given set of variable indices touches.
    pub fn blocks_touched_by(&self, var_indices: &[usize]) -> Vec<usize> {
        let mut seen = std::collections::BTreeSet::new();
        for &v in var_indices {
            if let Some(&b) = self.block_assignments.get(v) {
                seen.insert(b);
            }
        }
        seen.into_iter().collect()
    }

    /// Refine partition by splitting the largest block in two (balanced split).
    pub fn refine_largest_block(&self) -> CertificateResult<Self> {
        let (largest_idx, _) = self
            .block_sizes
            .iter()
            .enumerate()
            .max_by_key(|(_, &s)| s)
            .ok_or_else(|| CertificateError::invalid_partition("no blocks"))?;

        let vars_in_block = self.variables_in_block(largest_idx);
        if vars_in_block.len() < 2 {
            return Err(CertificateError::invalid_partition(
                "largest block has only 1 variable, cannot split",
            ));
        }

        let split_point = vars_in_block.len() / 2;
        let new_block_id = self.num_blocks;
        let new_num_blocks = self.num_blocks + 1;

        let mut new_assignments = self.block_assignments.clone();
        for &v in &vars_in_block[split_point..] {
            new_assignments[v] = new_block_id;
        }

        Partition::new(new_assignments, new_num_blocks, self.method.clone())
    }
}

/// L3 Partition-to-Bound Bridge Certificate.
///
/// Certifies the duality gap bound:
///
/// `z_LP − z_D(P) ≤ Σ_{e ∈ E_cross(P)} |y*_e| · (n_e(P) − 1)`
///
/// where `E_cross(P)` is the set of crossing constraints, `y*_e` is the
/// optimal dual value for constraint `e`, and `n_e(P)` is the number of
/// blocks that constraint `e` touches. Optionally records the actual LP and
/// decomposition objectives for tightness assessment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct L3PartitionCertificate {
    pub id: String,
    pub created_at: String,
    pub partition: Partition,
    pub crossing_edges: Vec<CrossingEdge>,
    pub dual_values: Vec<f64>,
    pub per_edge_contribution: Vec<f64>,
    pub total_bound: f64,
    pub verification_status: VerificationStatus,
    pub lp_objective: Option<f64>,
    pub decomp_objective: Option<f64>,
    pub actual_gap: Option<f64>,
    pub bound_is_tight: Option<bool>,
    pub metadata: IndexMap<String, String>,
}

/// Metadata for a single constraint, used to identify crossing edges.
///
/// Stores the constraint index, human-readable name, and the indices of
/// variables that appear in the constraint.
#[derive(Debug, Clone)]
pub struct ConstraintInfo {
    pub index: usize,
    pub name: String,
    pub variable_indices: Vec<usize>,
}

impl L3PartitionCertificate {
    /// Compute the L3 bound for a given partition and dual solution.
    ///
    /// The bound is: Σ_{e ∈ E_cross(P)} |y*_e| * (n_e(P) - 1)
    pub fn compute_bound(
        partition: &Partition,
        constraints: &[ConstraintInfo],
        dual_values: &[f64],
    ) -> CertificateResult<Self> {
        if dual_values.len() != constraints.len() {
            return Err(CertificateError::incomplete_data(
                "dual_values",
                format!(
                    "expected {} values, got {}",
                    constraints.len(),
                    dual_values.len()
                ),
            ));
        }

        for (i, &d) in dual_values.iter().enumerate() {
            if d.is_nan() || d.is_infinite() {
                return Err(CertificateError::numerical_precision(
                    format!("dual value {} is non-finite", i),
                    d,
                    0.0,
                ));
            }
        }

        let mut crossing_edges = Vec::new();
        let mut per_edge_contribution = Vec::new();
        let mut total_bound = 0.0;

        for (ci, constraint) in constraints.iter().enumerate() {
            let blocks_touched = partition.blocks_touched_by(&constraint.variable_indices);
            let num_blocks = blocks_touched.len();

            if num_blocks > 1 {
                let contribution = dual_values[ci].abs() * (num_blocks as f64 - 1.0);
                total_bound += contribution;
                per_edge_contribution.push(contribution);

                crossing_edges.push(CrossingEdge {
                    constraint_index: ci,
                    constraint_name: constraint.name.clone(),
                    blocks_touched,
                    num_blocks_touched: num_blocks,
                    dual_value: dual_values[ci],
                    contribution,
                });
            }
        }

        let id = Uuid::new_v4().to_string();
        let created_at = Utc::now().to_rfc3339();

        Ok(Self {
            id,
            created_at,
            partition: partition.clone(),
            crossing_edges,
            dual_values: dual_values.to_vec(),
            per_edge_contribution,
            total_bound,
            verification_status: VerificationStatus::Unverified,
            lp_objective: None,
            decomp_objective: None,
            actual_gap: None,
            bound_is_tight: None,
            metadata: IndexMap::new(),
        })
    }

    /// Verify the certificate by recomputing the bound and checking consistency.
    pub fn verify(&mut self) -> CertificateResult<bool> {
        let mut recomputed = 0.0;
        let mut issues = Vec::new();

        for edge in &self.crossing_edges {
            let expected_contribution =
                edge.dual_value.abs() * (edge.num_blocks_touched as f64 - 1.0);
            let diff = (expected_contribution - edge.contribution).abs();
            if diff > 1e-10 {
                issues.push(format!(
                    "edge {} contribution mismatch: expected {:.6e}, got {:.6e}",
                    edge.constraint_index, expected_contribution, edge.contribution
                ));
            }
            recomputed += expected_contribution;
        }

        let total_diff = (recomputed - self.total_bound).abs();
        if total_diff > 1e-10 {
            issues.push(format!(
                "total bound mismatch: recomputed={:.6e}, stored={:.6e}",
                recomputed, self.total_bound
            ));
        }

        // Check that crossing edges are correct w.r.t. partition
        for edge in &self.crossing_edges {
            if edge.num_blocks_touched < 2 {
                issues.push(format!(
                    "edge {} touches only {} blocks but is marked as crossing",
                    edge.constraint_index, edge.num_blocks_touched
                ));
            }
        }

        // Verify bound vs actual gap if available
        if let (Some(lp), Some(decomp)) = (self.lp_objective, self.decomp_objective) {
            let gap = (lp - decomp).abs();
            if gap > self.total_bound + 1e-8 {
                issues.push(format!(
                    "actual gap {:.6e} exceeds bound {:.6e}",
                    gap, self.total_bound
                ));
            }
            self.actual_gap = Some(gap);
            self.bound_is_tight = Some((gap - self.total_bound).abs() / (self.total_bound.abs() + 1e-15) < 0.1);
        }

        if issues.is_empty() {
            self.verification_status = VerificationStatus::Verified;
            Ok(true)
        } else {
            self.verification_status = VerificationStatus::Failed;
            Err(CertificateError::bound_verification_failed(
                issues.join("; "),
                self.total_bound,
                recomputed,
                1e-10,
            ))
        }
    }

    /// Try to improve (tighten) the bound via partition refinement.
    /// Splits the block that contributes most to the crossing weight.
    pub fn tighten(
        &self,
        constraints: &[ConstraintInfo],
    ) -> CertificateResult<L3PartitionCertificate> {
        // Identify which block contributes most to crossing edges
        let mut block_contribution = vec![0.0f64; self.partition.num_blocks];

        for edge in &self.crossing_edges {
            let share = edge.contribution / edge.num_blocks_touched as f64;
            for &b in &edge.blocks_touched {
                if b < block_contribution.len() {
                    block_contribution[b] += share;
                }
            }
        }

        // Find block with highest contribution and split it
        let (worst_block, _) = block_contribution
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or_else(|| CertificateError::invalid_partition("no blocks to refine"))?;

        let vars_in_worst = self.partition.variables_in_block(worst_block);
        if vars_in_worst.len() < 2 {
            return Err(CertificateError::invalid_partition(
                "worst contributing block has only 1 variable",
            ));
        }

        let split_point = vars_in_worst.len() / 2;
        let new_block_id = self.partition.num_blocks;
        let new_num_blocks = self.partition.num_blocks + 1;

        let mut new_assignments = self.partition.block_assignments.clone();
        for &v in &vars_in_worst[split_point..] {
            new_assignments[v] = new_block_id;
        }

        let new_partition =
            Partition::new(new_assignments, new_num_blocks, self.partition.method.clone())?;

        Self::compute_bound(&new_partition, constraints, &self.dual_values)
    }

    /// Top N edges contributing most to the bound.
    pub fn top_contributors(&self, n: usize) -> Vec<&CrossingEdge> {
        let mut sorted: Vec<&CrossingEdge> = self.crossing_edges.iter().collect();
        sorted.sort_by(|a, b| {
            b.contribution
                .partial_cmp(&a.contribution)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted.truncate(n);
        sorted
    }

    /// Fraction of the total bound from the top N contributors.
    pub fn concentration_ratio(&self, n: usize) -> f64 {
        if self.total_bound.abs() < 1e-15 {
            return 0.0;
        }
        let top_sum: f64 = self.top_contributors(n).iter().map(|e| e.contribution).sum();
        top_sum / self.total_bound
    }

    /// Average contribution per crossing edge.
    pub fn average_contribution(&self) -> f64 {
        if self.crossing_edges.is_empty() {
            return 0.0;
        }
        self.total_bound / self.crossing_edges.len() as f64
    }

    /// Ratio of crossing edges to total constraints.
    pub fn crossing_ratio(&self, total_constraints: usize) -> f64 {
        if total_constraints == 0 {
            return 0.0;
        }
        self.crossing_edges.len() as f64 / total_constraints as f64
    }

    /// Set known objective values for gap validation.
    pub fn set_objectives(&mut self, lp_obj: f64, decomp_obj: f64) {
        self.lp_objective = Some(lp_obj);
        self.decomp_objective = Some(decomp_obj);
        self.actual_gap = Some((lp_obj - decomp_obj).abs());
        self.verification_status = VerificationStatus::Stale;
    }

    /// Add metadata to the certificate.
    pub fn add_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }

    /// Summary statistics as a map.
    pub fn summary_stats(&self) -> IndexMap<String, f64> {
        let mut stats = IndexMap::new();
        stats.insert("total_bound".to_string(), self.total_bound);
        stats.insert(
            "num_crossing_edges".to_string(),
            self.crossing_edges.len() as f64,
        );
        stats.insert("num_blocks".to_string(), self.partition.num_blocks as f64);
        stats.insert(
            "num_variables".to_string(),
            self.partition.num_variables as f64,
        );
        stats.insert("balance_ratio".to_string(), self.partition.balance_ratio());
        stats.insert("avg_contribution".to_string(), self.average_contribution());
        if let Some(gap) = self.actual_gap {
            stats.insert("actual_gap".to_string(), gap);
            if self.total_bound > 1e-15 {
                stats.insert("tightness_ratio".to_string(), gap / self.total_bound);
            }
        }
        stats
    }

    /// Create a certificate for a trivially decomposable problem (no crossing edges).
    pub fn trivial(partition: &Partition) -> Self {
        let id = Uuid::new_v4().to_string();
        Self {
            id,
            created_at: Utc::now().to_rfc3339(),
            partition: partition.clone(),
            crossing_edges: Vec::new(),
            dual_values: Vec::new(),
            per_edge_contribution: Vec::new(),
            total_bound: 0.0,
            verification_status: VerificationStatus::Verified,
            lp_objective: None,
            decomp_objective: None,
            actual_gap: None,
            bound_is_tight: Some(true),
            metadata: IndexMap::new(),
        }
    }
}

#[cfg(test)]
fn make_test_partition(n: usize, k: usize) -> Partition {
    let assignments: Vec<usize> = (0..n).map(|i| i % k).collect();
    Partition::new(assignments, k, PartitionMethod::Manual).unwrap()
}

#[cfg(test)]
fn make_test_constraints(n: usize, k: usize) -> Vec<ConstraintInfo> {
    let mut constraints = Vec::new();
    // Internal constraints (within one block)
    for block in 0..k {
        let start = block;
        let end = if start + k < n { start + k } else { start };
        constraints.push(ConstraintInfo {
            index: constraints.len(),
            name: format!("internal_b{}_{}", block, constraints.len()),
            variable_indices: vec![start, end],
        });
    }
    // Crossing constraints (across blocks)
    for i in 0..k {
        let j = (i + 1) % k;
        let v1 = i;
        let v2 = if j < n { j } else { 0 };
        constraints.push(ConstraintInfo {
            index: constraints.len(),
            name: format!("crossing_{}_{}", i, j),
            variable_indices: vec![v1, v2],
        });
    }
    constraints
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partition_creation() {
        let p = Partition::new(vec![0, 0, 1, 1, 2, 2], 3, PartitionMethod::Spectral).unwrap();
        assert_eq!(p.num_blocks, 3);
        assert_eq!(p.num_variables, 6);
        assert_eq!(p.block_sizes, vec![2, 2, 2]);
    }

    #[test]
    fn test_partition_empty_fails() {
        let result = Partition::new(vec![], 1, PartitionMethod::Manual);
        assert!(result.is_err());
    }

    #[test]
    fn test_partition_empty_block_fails() {
        let result = Partition::new(vec![0, 0, 2, 2], 3, PartitionMethod::Manual);
        assert!(result.is_err());
    }

    #[test]
    fn test_partition_block_of() {
        let p = Partition::new(vec![0, 1, 2, 0, 1, 2], 3, PartitionMethod::GCG).unwrap();
        assert_eq!(p.block_of(0), Some(0));
        assert_eq!(p.block_of(1), Some(1));
        assert_eq!(p.block_of(2), Some(2));
        assert_eq!(p.block_of(6), None);
    }

    #[test]
    fn test_partition_variables_in_block() {
        let p = Partition::new(vec![0, 1, 0, 1, 0], 2, PartitionMethod::KMeans).unwrap();
        assert_eq!(p.variables_in_block(0), vec![0, 2, 4]);
        assert_eq!(p.variables_in_block(1), vec![1, 3]);
    }

    #[test]
    fn test_partition_balance_ratio() {
        let balanced = Partition::new(vec![0, 1, 0, 1], 2, PartitionMethod::Manual).unwrap();
        assert!((balanced.balance_ratio() - 1.0).abs() < 1e-10);

        let imbalanced = Partition::new(vec![0, 0, 0, 1], 2, PartitionMethod::Manual).unwrap();
        assert!((imbalanced.balance_ratio() - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_partition_refine() {
        let p = Partition::new(vec![0, 0, 0, 0, 1, 1], 2, PartitionMethod::Manual).unwrap();
        let refined = p.refine_largest_block().unwrap();
        assert_eq!(refined.num_blocks, 3);
        assert_eq!(refined.num_variables, 6);
    }

    #[test]
    fn test_crossing_edge_contribution() {
        let mut edge = CrossingEdge {
            constraint_index: 0,
            constraint_name: "c0".to_string(),
            blocks_touched: vec![0, 1, 2],
            num_blocks_touched: 3,
            dual_value: -2.5,
            contribution: 0.0,
        };
        edge.compute_contribution();
        assert!((edge.contribution - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_bound_simple() {
        let partition = make_test_partition(6, 3);
        let constraints = vec![
            ConstraintInfo {
                index: 0,
                name: "c0".into(),
                variable_indices: vec![0, 1],
            },
            ConstraintInfo {
                index: 1,
                name: "c1".into(),
                variable_indices: vec![0, 3],
            },
        ];
        let duals = vec![1.0, 2.0];
        let cert = L3PartitionCertificate::compute_bound(&partition, &constraints, &duals).unwrap();
        // c0 touches blocks 0,1 → contribution = 1.0 * 1 = 1.0
        // c1 touches block 0 only (vars 0 and 3 are both block 0) → not crossing
        assert!(!cert.crossing_edges.is_empty());
        assert!(cert.total_bound > 0.0);
    }

    #[test]
    fn test_compute_bound_no_crossing() {
        let partition = Partition::new(vec![0, 0, 1, 1], 2, PartitionMethod::Manual).unwrap();
        let constraints = vec![
            ConstraintInfo {
                index: 0,
                name: "internal0".into(),
                variable_indices: vec![0, 1],
            },
            ConstraintInfo {
                index: 1,
                name: "internal1".into(),
                variable_indices: vec![2, 3],
            },
        ];
        let duals = vec![5.0, 3.0];
        let cert =
            L3PartitionCertificate::compute_bound(&partition, &constraints, &duals).unwrap();
        assert_eq!(cert.crossing_edges.len(), 0);
        assert!((cert.total_bound - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_verify_valid() {
        let partition = make_test_partition(6, 2);
        let constraints = vec![ConstraintInfo {
            index: 0,
            name: "cross".into(),
            variable_indices: vec![0, 1],
        }];
        let duals = vec![3.0];
        let mut cert =
            L3PartitionCertificate::compute_bound(&partition, &constraints, &duals).unwrap();
        let result = cert.verify();
        assert!(result.is_ok());
        assert_eq!(cert.verification_status, VerificationStatus::Verified);
    }

    #[test]
    fn test_verify_with_objectives() {
        let partition = make_test_partition(6, 2);
        let constraints = vec![ConstraintInfo {
            index: 0,
            name: "cross".into(),
            variable_indices: vec![0, 1],
        }];
        let duals = vec![3.0];
        let mut cert =
            L3PartitionCertificate::compute_bound(&partition, &constraints, &duals).unwrap();
        cert.set_objectives(100.0, 98.0);
        let _ = cert.verify();
        assert!(cert.actual_gap.is_some());
    }

    #[test]
    fn test_tighten() {
        let partition = make_test_partition(8, 2);
        let constraints = vec![ConstraintInfo {
            index: 0,
            name: "cross".into(),
            variable_indices: vec![0, 1, 2, 3],
        }];
        let duals = vec![5.0];
        let cert =
            L3PartitionCertificate::compute_bound(&partition, &constraints, &duals).unwrap();
        let tightened = cert.tighten(&constraints).unwrap();
        assert!(tightened.partition.num_blocks > partition.num_blocks);
    }

    #[test]
    fn test_top_contributors() {
        let partition = make_test_partition(6, 3);
        let constraints = make_test_constraints(6, 3);
        let duals: Vec<f64> = (0..constraints.len()).map(|i| (i + 1) as f64).collect();
        let cert =
            L3PartitionCertificate::compute_bound(&partition, &constraints, &duals).unwrap();
        let top = cert.top_contributors(2);
        assert!(top.len() <= 2);
        if top.len() >= 2 {
            assert!(top[0].contribution >= top[1].contribution);
        }
    }

    #[test]
    fn test_concentration_ratio() {
        let partition = make_test_partition(6, 3);
        let constraints = make_test_constraints(6, 3);
        let duals: Vec<f64> = (0..constraints.len()).map(|i| (i + 1) as f64 * 0.5).collect();
        let cert =
            L3PartitionCertificate::compute_bound(&partition, &constraints, &duals).unwrap();
        if !cert.crossing_edges.is_empty() {
            let ratio = cert.concentration_ratio(cert.crossing_edges.len());
            assert!((ratio - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_trivial_certificate() {
        let partition = make_test_partition(4, 2);
        let cert = L3PartitionCertificate::trivial(&partition);
        assert_eq!(cert.total_bound, 0.0);
        assert_eq!(cert.verification_status, VerificationStatus::Verified);
    }

    #[test]
    fn test_summary_stats() {
        let partition = make_test_partition(6, 2);
        let constraints = vec![ConstraintInfo {
            index: 0,
            name: "c".into(),
            variable_indices: vec![0, 1],
        }];
        let duals = vec![2.0];
        let cert =
            L3PartitionCertificate::compute_bound(&partition, &constraints, &duals).unwrap();
        let stats = cert.summary_stats();
        assert!(stats.contains_key("total_bound"));
        assert!(stats.contains_key("num_blocks"));
    }

    #[test]
    fn test_metadata() {
        let partition = make_test_partition(4, 2);
        let mut cert = L3PartitionCertificate::trivial(&partition);
        cert.add_metadata("solver", "CPLEX");
        assert_eq!(cert.metadata.get("solver").unwrap(), "CPLEX");
    }
}
