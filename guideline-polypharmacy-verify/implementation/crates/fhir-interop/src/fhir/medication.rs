//! FHIR R4 MedicationRequest and MedicationStatement → internal Medication.

use serde::{Deserialize, Serialize};

use guardpharma_clinical::medication::{Medication, MedicationStatus};
use guardpharma_types::{DrugId, DrugRoute, DosingSchedule};

use super::types::{
    CodeableConcept, Dosage, Meta, Reference,
    text_to_drug_route, snomed_route_to_drug_route,
};
use crate::error::FhirInteropError;

/// FHIR R4 MedicationRequest resource.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FhirMedicationRequest {
    #[serde(rename = "resourceType")]
    pub resource_type: String,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub meta: Option<Meta>,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub intent: Option<String>,
    #[serde(rename = "medicationCodeableConcept", default)]
    pub medication_codeable_concept: Option<CodeableConcept>,
    #[serde(rename = "medicationReference", default)]
    pub medication_reference: Option<Reference>,
    #[serde(default)]
    pub subject: Option<Reference>,
    #[serde(rename = "dosageInstruction", default)]
    pub dosage_instruction: Vec<Dosage>,
    #[serde(rename = "dispenseRequest", default)]
    pub dispense_request: Option<serde_json::Value>,
    #[serde(rename = "reasonCode", default)]
    pub reason_code: Vec<CodeableConcept>,
}

impl FhirMedicationRequest {
    /// Parse from a JSON string.
    pub fn from_json(json: &str) -> Result<Self, FhirInteropError> {
        let res: Self = serde_json::from_str(json)?;
        if res.resource_type != "MedicationRequest" {
            return Err(FhirInteropError::ResourceTypeMismatch {
                expected: "MedicationRequest".into(),
                got: res.resource_type,
            });
        }
        Ok(res)
    }

    /// Convert to the internal Medication type.
    pub fn to_medication(&self) -> Result<Medication, FhirInteropError> {
        let (drug_id, name) = extract_drug_identity(
            self.medication_codeable_concept.as_ref(),
            self.medication_reference.as_ref(),
        )?;

        let status = fhir_status_to_medication_status(self.status.as_deref());

        let (dose_mg, route, interval_hours) = extract_dosing(&self.dosage_instruction);

        let doses_per_day = if interval_hours > 0.0 {
            (24.0 / interval_hours).ceil() as u32
        } else {
            1
        };

        let indication = self
            .reason_code
            .first()
            .and_then(|cc| cc.display_text())
            .unwrap_or("")
            .to_string();

        Ok(Medication {
            drug_id,
            name,
            dose_mg,
            schedule: DosingSchedule::new(interval_hours, dose_mg, route.clone(), doses_per_day),
            route,
            indication,
            status,
            essential: false,
            classification: None,
        })
    }
}

/// FHIR R4 MedicationStatement resource.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FhirMedicationStatement {
    #[serde(rename = "resourceType")]
    pub resource_type: String,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub meta: Option<Meta>,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(rename = "medicationCodeableConcept", default)]
    pub medication_codeable_concept: Option<CodeableConcept>,
    #[serde(rename = "medicationReference", default)]
    pub medication_reference: Option<Reference>,
    #[serde(default)]
    pub subject: Option<Reference>,
    #[serde(default)]
    pub dosage: Vec<Dosage>,
    #[serde(rename = "reasonCode", default)]
    pub reason_code: Vec<CodeableConcept>,
}

impl FhirMedicationStatement {
    /// Parse from a JSON string.
    pub fn from_json(json: &str) -> Result<Self, FhirInteropError> {
        let res: Self = serde_json::from_str(json)?;
        if res.resource_type != "MedicationStatement" {
            return Err(FhirInteropError::ResourceTypeMismatch {
                expected: "MedicationStatement".into(),
                got: res.resource_type,
            });
        }
        Ok(res)
    }

    /// Convert to the internal Medication type.
    pub fn to_medication(&self) -> Result<Medication, FhirInteropError> {
        let (drug_id, name) = extract_drug_identity(
            self.medication_codeable_concept.as_ref(),
            self.medication_reference.as_ref(),
        )?;

        let status = match self.status.as_deref() {
            Some("active") | Some("intended") => MedicationStatus::Active,
            Some("completed") => MedicationStatus::Completed,
            Some("on-hold") => MedicationStatus::OnHold,
            Some("stopped") | Some("not-taken") => MedicationStatus::Discontinued,
            _ => MedicationStatus::Active,
        };

        let (dose_mg, route, interval_hours) = extract_dosing(&self.dosage);

        let doses_per_day = if interval_hours > 0.0 {
            (24.0 / interval_hours).ceil() as u32
        } else {
            1
        };

        let indication = self
            .reason_code
            .first()
            .and_then(|cc| cc.display_text())
            .unwrap_or("")
            .to_string();

        Ok(Medication {
            drug_id,
            name,
            dose_mg,
            schedule: DosingSchedule::new(interval_hours, dose_mg, route.clone(), doses_per_day),
            route,
            indication,
            status,
            essential: false,
            classification: None,
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Extract drug identity from either a CodeableConcept or a Reference.
fn extract_drug_identity(
    codeable: Option<&CodeableConcept>,
    reference: Option<&Reference>,
) -> Result<(DrugId, String), FhirInteropError> {
    if let Some(cc) = codeable {
        let code = cc
            .rxnorm_code()
            .or_else(|| cc.coding.first().and_then(|c| c.code.as_deref()));
        let display = cc.display_text().unwrap_or("Unknown");
        let id_str = code.unwrap_or(display);
        return Ok((DrugId::new(id_str), display.to_string()));
    }

    if let Some(r) = reference {
        let display = r.display.as_deref().unwrap_or("Unknown");
        let id = r.resource_id().unwrap_or(display);
        return Ok((DrugId::new(id), display.to_string()));
    }

    Err(FhirInteropError::MissingField("medication".into()))
}

/// Extract dose, route, and interval from a list of FHIR Dosage elements.
fn extract_dosing(dosage: &[Dosage]) -> (f64, DrugRoute, f64) {
    let d = match dosage.first() {
        Some(d) => d,
        None => return (0.0, DrugRoute::Oral, 24.0),
    };

    let dose_mg = d
        .dose_and_rate
        .first()
        .and_then(|dr| dr.dose_quantity.as_ref())
        .and_then(|q| q.value)
        .unwrap_or(0.0);

    let route = d
        .route
        .as_ref()
        .and_then(|r| {
            // Try SNOMED code first, then display text
            r.coding
                .first()
                .and_then(|c| c.code.as_deref())
                .map(snomed_route_to_drug_route)
                .or_else(|| r.display_text().map(text_to_drug_route))
        })
        .unwrap_or(DrugRoute::Oral);

    let interval_hours = d
        .timing
        .as_ref()
        .and_then(|t| t.repeat.as_ref())
        .and_then(|r| r.interval_hours())
        .unwrap_or(24.0);

    (dose_mg, route, interval_hours)
}

/// Map a FHIR MedicationRequest status to the internal MedicationStatus.
fn fhir_status_to_medication_status(status: Option<&str>) -> MedicationStatus {
    match status {
        Some("active") | Some("draft") => MedicationStatus::Active,
        Some("on-hold") => MedicationStatus::OnHold,
        Some("completed") => MedicationStatus::Completed,
        Some("cancelled") | Some("entered-in-error") => MedicationStatus::Cancelled,
        Some("stopped") => MedicationStatus::Discontinued,
        _ => MedicationStatus::Active,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_medication_request() {
        let json = r#"{
            "resourceType": "MedicationRequest",
            "id": "medrx-1",
            "status": "active",
            "intent": "order",
            "medicationCodeableConcept": {
                "coding": [{
                    "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                    "code": "860975",
                    "display": "Metformin 500 MG Oral Tablet"
                }],
                "text": "Metformin 500mg"
            },
            "subject": {"reference": "Patient/123"},
            "dosageInstruction": [{
                "timing": {
                    "repeat": {"frequency": 2, "period": 1.0, "periodUnit": "d"}
                },
                "route": {
                    "coding": [{"system": "http://snomed.info/sct", "code": "26643006", "display": "Oral"}]
                },
                "doseAndRate": [{
                    "doseQuantity": {"value": 500.0, "unit": "mg"}
                }]
            }]
        }"#;
        let mr = FhirMedicationRequest::from_json(json).unwrap();
        let med = mr.to_medication().unwrap();
        assert_eq!(med.name, "Metformin 500 MG Oral Tablet");
        assert!((med.dose_mg - 500.0).abs() < f64::EPSILON);
        assert_eq!(med.route, DrugRoute::Oral);
        assert_eq!(med.status, MedicationStatus::Active);
    }

    #[test]
    fn parse_medication_statement() {
        let json = r#"{
            "resourceType": "MedicationStatement",
            "id": "medstmt-1",
            "status": "active",
            "medicationCodeableConcept": {
                "coding": [{
                    "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                    "code": "197361",
                    "display": "Amlodipine 5 MG Oral Tablet"
                }]
            },
            "subject": {"reference": "Patient/456"},
            "dosage": [{
                "doseAndRate": [{
                    "doseQuantity": {"value": 5.0, "unit": "mg"}
                }]
            }]
        }"#;
        let ms = FhirMedicationStatement::from_json(json).unwrap();
        let med = ms.to_medication().unwrap();
        assert_eq!(med.name, "Amlodipine 5 MG Oral Tablet");
        assert!((med.dose_mg - 5.0).abs() < f64::EPSILON);
    }
}
