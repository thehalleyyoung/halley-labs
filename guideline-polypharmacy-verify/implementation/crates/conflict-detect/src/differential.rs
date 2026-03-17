//! Differential analysis for medication changes.
//!
//! When a patient's medication list changes (add, remove, dose adjust), the
//! differential analyzer determines which verification results are still valid
//! and which must be re-evaluated — avoiding a full re-verification of the
//! entire medication set.

use serde::{Deserialize, Serialize};

use crate::types::{
    ConfirmedConflict, ConflictSeverity, DrugId, MedicationRecord,
    PatientProfile, SafetyVerdict,
};

// ---------------------------------------------------------------------------
// Change representation
// ---------------------------------------------------------------------------

/// A single medication change.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MedicationChange {
    /// A new drug was added.
    Added(MedicationRecord),
    /// A drug was removed.
    Removed(DrugId),
    /// A drug's dosage was modified.
    DoseAdjusted {
        drug_id: DrugId,
        old: MedicationRecord,
        new: MedicationRecord,
    },
}

// ---------------------------------------------------------------------------
// DifferentialAnalyzer
// ---------------------------------------------------------------------------

/// Computes the delta between two medication states and determines which
/// interactions need re-verification.
#[derive(Debug, Clone, Default)]
pub struct DifferentialAnalyzer;

impl DifferentialAnalyzer {
    /// Analyze the difference between an old and new medication list.
    pub fn analyze(
        &self,
        old_meds: &[MedicationRecord],
        new_meds: &[MedicationRecord],
    ) -> DifferentialResult {
        let mut changes = Vec::new();

        let old_ids: std::collections::HashSet<&DrugId> =
            old_meds.iter().map(|m| &m.drug.id).collect();
        let new_ids: std::collections::HashSet<&DrugId> =
            new_meds.iter().map(|m| &m.drug.id).collect();

        // Additions
        for med in new_meds {
            if !old_ids.contains(&med.drug.id) {
                changes.push(MedicationChange::Added(med.clone()));
            }
        }
        // Removals
        for med in old_meds {
            if !new_ids.contains(&med.drug.id) {
                changes.push(MedicationChange::Removed(med.drug.id.clone()));
            }
        }

        let affected_drugs: Vec<DrugId> = changes
            .iter()
            .map(|c| match c {
                MedicationChange::Added(m) => m.drug.id.clone(),
                MedicationChange::Removed(id) => id.clone(),
                MedicationChange::DoseAdjusted { drug_id, .. } => drug_id.clone(),
            })
            .collect();

        DifferentialResult {
            changes,
            affected_drugs,
            requires_full_reverify: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Result of a differential analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DifferentialResult {
    /// List of detected medication changes.
    pub changes: Vec<MedicationChange>,
    /// Drug IDs that are affected and need re-verification.
    pub affected_drugs: Vec<DrugId>,
    /// Whether a full re-verification is required (e.g. too many changes).
    pub requires_full_reverify: bool,
}

/// Human-readable differential report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DifferentialReport {
    pub summary: String,
    pub change_count: usize,
    pub new_interactions_possible: usize,
    pub removed_interactions: usize,
}

impl DifferentialReport {
    /// Create a report from a differential result.
    pub fn from_result(result: &DifferentialResult) -> Self {
        let added = result
            .changes
            .iter()
            .filter(|c| matches!(c, MedicationChange::Added(_)))
            .count();
        let removed = result
            .changes
            .iter()
            .filter(|c| matches!(c, MedicationChange::Removed(_)))
            .count();
        Self {
            summary: format!(
                "{} medication change(s): {} added, {} removed",
                result.changes.len(),
                added,
                removed
            ),
            change_count: result.changes.len(),
            new_interactions_possible: added,
            removed_interactions: removed,
        }
    }
}
