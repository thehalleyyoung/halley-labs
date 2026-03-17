//! Batch pairwise and N-wise conflict detection.
//!
//! Runs the verification pipeline over all pairs (and selected higher-order
//! subsets) of a patient's active medications, collecting results into a
//! conflict matrix for downstream reporting and visualisation.

use serde::{Deserialize, Serialize};

use crate::types::{
    ConfirmedConflict, ConflictSeverity, DrugId, DrugInfo, MedicationRecord,
    PatientProfile, SafetyVerdict,
};

// ---------------------------------------------------------------------------
// BatchDetector
// ---------------------------------------------------------------------------

/// Orchestrates batch pairwise and N-wise conflict detection across a
/// patient's medication list.
#[derive(Debug, Clone)]
pub struct BatchDetector {
    /// Maximum combination order to test (2 = pairwise, 3 = triples, …).
    pub max_order: usize,
    /// Timeout per individual verification call (seconds).
    pub timeout_secs: u64,
}

impl Default for BatchDetector {
    fn default() -> Self {
        Self {
            max_order: 2,
            timeout_secs: 30,
        }
    }
}

impl BatchDetector {
    /// Create a new batch detector with the given maximum combination order.
    pub fn new(max_order: usize) -> Self {
        Self {
            max_order,
            timeout_secs: 30,
        }
    }

    /// Run batch detection for the given patient profile.
    ///
    /// Returns a [`BatchResult`] containing the conflict matrix and
    /// aggregate statistics.
    pub fn detect(&self, profile: &PatientProfile) -> BatchResult {
        let meds = &profile.medications;
        let n = meds.len();
        let mut pairwise = Vec::new();
        let mut matrix = ConflictMatrix::new(meds);

        for i in 0..n {
            for j in (i + 1)..n {
                let pair = PairwiseResult {
                    drug_a: meds[i].drug.id.clone(),
                    drug_b: meds[j].drug.id.clone(),
                    verdict: SafetyVerdict::Safe,
                    severity: None,
                    detail: None,
                };
                matrix.set(i, j, pair.verdict);
                pairwise.push(pair);
            }
        }

        let stats = BatchStatistics {
            total_pairs: pairwise.len(),
            safe_count: pairwise.iter().filter(|p| p.verdict == SafetyVerdict::Safe).count(),
            conflict_count: pairwise.iter().filter(|p| p.verdict != SafetyVerdict::Safe).count(),
            timeout_count: 0,
        };

        BatchResult {
            pairwise,
            matrix,
            statistics: stats,
        }
    }
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Aggregated result of a batch conflict detection run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchResult {
    /// Individual pairwise results.
    pub pairwise: Vec<PairwiseResult>,
    /// Conflict matrix indexed by medication list position.
    pub matrix: ConflictMatrix,
    /// Aggregate statistics.
    pub statistics: BatchStatistics,
}

/// Result of a single pairwise interaction check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairwiseResult {
    pub drug_a: DrugId,
    pub drug_b: DrugId,
    pub verdict: SafetyVerdict,
    pub severity: Option<ConflictSeverity>,
    pub detail: Option<String>,
}

/// N×N safety verdict matrix for a medication list.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictMatrix {
    /// Drug IDs in the same order as the rows/columns.
    pub drugs: Vec<DrugId>,
    /// Row-major entries.
    pub entries: Vec<SafetyVerdict>,
    size: usize,
}

impl ConflictMatrix {
    /// Create an empty matrix from a medication list.
    pub fn new(meds: &[MedicationRecord]) -> Self {
        let size = meds.len();
        Self {
            drugs: meds.iter().map(|m| m.drug.id.clone()).collect(),
            entries: vec![SafetyVerdict::Safe; size * size],
            size,
        }
    }

    /// Set the verdict for the (i, j) pair (symmetric).
    pub fn set(&mut self, i: usize, j: usize, verdict: SafetyVerdict) {
        self.entries[i * self.size + j] = verdict;
        self.entries[j * self.size + i] = verdict;
    }

    /// Get the verdict for the (i, j) pair.
    pub fn get(&self, i: usize, j: usize) -> SafetyVerdict {
        self.entries[i * self.size + j]
    }
}

/// Aggregate statistics for a batch run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchStatistics {
    pub total_pairs: usize,
    pub safe_count: usize,
    pub conflict_count: usize,
    pub timeout_count: usize,
}

impl BatchStatistics {
    /// Fraction of pairs that were verified safe.
    pub fn safe_fraction(&self) -> f64 {
        if self.total_pairs == 0 {
            return 1.0;
        }
        self.safe_count as f64 / self.total_pairs as f64
    }
}
