//! Medication management, reconciliation, and history.

use serde::{Deserialize, Serialize};
use guardpharma_types::{DrugId, DrugRoute, DosingSchedule, DrugClass};

/// Status of a medication order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MedicationStatus {
    Active,
    OnHold,
    Completed,
    Cancelled,
    Discontinued,
}

/// Drug classification for formulary purposes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrugClassification {
    pub atc_code: Option<String>,
    pub therapeutic_class: DrugClass,
    pub pharmacologic_class: String,
}

/// An active medication in the patient's regimen.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Medication {
    pub drug_id: DrugId,
    pub name: String,
    pub dose_mg: f64,
    pub schedule: DosingSchedule,
    pub route: DrugRoute,
    pub indication: String,
    pub status: MedicationStatus,
    pub essential: bool,
    pub classification: Option<DrugClassification>,
}

impl Medication {
    pub fn new(drug_id: DrugId, name: &str, dose_mg: f64, interval_hours: f64) -> Self {
        Medication {
            drug_id,
            name: name.to_string(),
            dose_mg,
            schedule: DosingSchedule::new(interval_hours, dose_mg, DrugRoute::Oral, (24.0 / interval_hours).ceil() as u32),
            route: DrugRoute::Oral,
            indication: String::new(),
            status: MedicationStatus::Active,
            essential: false,
            classification: None,
        }
    }

    pub fn with_indication(mut self, indication: &str) -> Self {
        self.indication = indication.to_string();
        self
    }
}

/// List of medications with utility methods.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MedicationList {
    pub medications: Vec<Medication>,
}

impl MedicationList {
    pub fn new() -> Self { MedicationList { medications: Vec::new() } }

    pub fn add(&mut self, med: Medication) { self.medications.push(med); }

    pub fn active(&self) -> Vec<&Medication> {
        self.medications.iter().filter(|m| m.status == MedicationStatus::Active).collect()
    }

    pub fn count(&self) -> usize { self.medications.len() }
}

/// Type of medication change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MedicationChangeType {
    Added,
    Removed,
    DoseChanged,
    FrequencyChanged,
    RouteChanged,
    Substituted,
}

/// A change to a medication.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MedicationChange {
    pub drug_id: DrugId,
    pub change_type: MedicationChangeType,
    pub old_value: Option<String>,
    pub new_value: Option<String>,
    pub reason: String,
}

/// Medication history for a patient.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MedicationHistory {
    pub changes: Vec<MedicationChange>,
}

/// Medication reconciliation result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MedicationReconciliation {
    pub reconciled_list: MedicationList,
    pub discrepancies: Vec<ReconciliationDiscrepancy>,
}

/// Type of reconciliation discrepancy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DiscrepancyType {
    Omission,
    Commission,
    DoseDifference,
    FrequencyDifference,
    Duplication,
}

/// A discrepancy found during reconciliation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReconciliationDiscrepancy {
    pub drug_id: DrugId,
    pub discrepancy_type: DiscrepancyType,
    pub description: String,
}
