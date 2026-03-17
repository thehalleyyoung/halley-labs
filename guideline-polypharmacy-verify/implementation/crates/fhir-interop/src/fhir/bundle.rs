//! FHIR R4 Bundle resource parsing.

use serde::{Deserialize, Serialize};

use guardpharma_clinical::PatientProfile;

use super::medication::{FhirMedicationRequest, FhirMedicationStatement};
use super::patient::{FhirPatientResource, FhirConditionResource, FhirObservationResource};
use super::plan_definition::FhirPlanDefinition;
use super::types::Meta;
use crate::error::FhirInteropError;

/// A single entry within a FHIR Bundle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleEntry {
    #[serde(rename = "fullUrl", default)]
    pub full_url: Option<String>,
    /// The embedded resource as a raw JSON value so we can dispatch on resourceType.
    pub resource: serde_json::Value,
}

/// FHIR R4 Bundle resource.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FhirBundle {
    #[serde(rename = "resourceType")]
    pub resource_type: String,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub meta: Option<Meta>,
    #[serde(rename = "type")]
    pub bundle_type: String,
    #[serde(default)]
    pub entry: Vec<BundleEntry>,
}

/// Parsed results from a FHIR Bundle.
#[derive(Debug, Clone, Default)]
pub struct ParsedBundle {
    pub patients: Vec<FhirPatientResource>,
    pub conditions: Vec<FhirConditionResource>,
    pub observations: Vec<FhirObservationResource>,
    pub medication_requests: Vec<FhirMedicationRequest>,
    pub medication_statements: Vec<FhirMedicationStatement>,
    pub plan_definitions: Vec<FhirPlanDefinition>,
    pub unrecognized_types: Vec<String>,
}

impl FhirBundle {
    /// Parse from a JSON string.
    pub fn from_json(json: &str) -> Result<Self, FhirInteropError> {
        let res: Self = serde_json::from_str(json)?;
        if res.resource_type != "Bundle" {
            return Err(FhirInteropError::ResourceTypeMismatch {
                expected: "Bundle".into(),
                got: res.resource_type,
            });
        }
        Ok(res)
    }

    /// Dispatch each entry by its resourceType and parse into typed structs.
    pub fn parse_entries(&self) -> ParsedBundle {
        let mut result = ParsedBundle::default();

        for entry in &self.entry {
            let rt = entry
                .resource
                .get("resourceType")
                .and_then(|v| v.as_str())
                .unwrap_or("");

            let json_str = match serde_json::to_string(&entry.resource) {
                Ok(s) => s,
                Err(e) => {
                    log::warn!("Failed to serialize bundle entry: {e}");
                    continue;
                }
            };

            match rt {
                "Patient" => match serde_json::from_str(&json_str) {
                    Ok(p) => result.patients.push(p),
                    Err(e) => log::warn!("Failed to parse Patient: {e}"),
                },
                "Condition" => match serde_json::from_str(&json_str) {
                    Ok(c) => result.conditions.push(c),
                    Err(e) => log::warn!("Failed to parse Condition: {e}"),
                },
                "Observation" => match serde_json::from_str(&json_str) {
                    Ok(o) => result.observations.push(o),
                    Err(e) => log::warn!("Failed to parse Observation: {e}"),
                },
                "MedicationRequest" => match serde_json::from_str(&json_str) {
                    Ok(mr) => result.medication_requests.push(mr),
                    Err(e) => log::warn!("Failed to parse MedicationRequest: {e}"),
                },
                "MedicationStatement" => match serde_json::from_str(&json_str) {
                    Ok(ms) => result.medication_statements.push(ms),
                    Err(e) => log::warn!("Failed to parse MedicationStatement: {e}"),
                },
                "PlanDefinition" => match serde_json::from_str(&json_str) {
                    Ok(pd) => result.plan_definitions.push(pd),
                    Err(e) => log::warn!("Failed to parse PlanDefinition: {e}"),
                },
                other if !other.is_empty() => {
                    result.unrecognized_types.push(other.to_string());
                }
                _ => {}
            }
        }

        result
    }

    /// Convenience: parse a Bundle and assemble a PatientProfile from all
    /// contained resources for the first Patient found.
    pub fn to_patient_profile(&self) -> Result<PatientProfile, FhirInteropError> {
        let parsed = self.parse_entries();

        let patient_resource = parsed
            .patients
            .first()
            .ok_or_else(|| FhirInteropError::MissingField("Patient resource in Bundle".into()))?;

        let mut profile = patient_resource.to_patient_profile()?;

        // Add conditions
        for cond in &parsed.conditions {
            profile.conditions.push(cond.to_patient_condition());
        }

        // Add lab values from observations
        for obs in &parsed.observations {
            if let Some(lv) = obs.to_lab_value() {
                profile.lab_values.push(lv);
            }
        }

        // Add medications
        for mr in &parsed.medication_requests {
            if let Ok(med) = mr.to_medication() {
                profile.current_medications.push(med.drug_id);
            }
        }
        for ms in &parsed.medication_statements {
            if let Ok(med) = ms.to_medication() {
                profile.current_medications.push(med.drug_id);
            }
        }

        Ok(profile)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_bundle_with_patient_and_meds() {
        let json = r#"{
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Patient",
                        "id": "pt-1",
                        "gender": "female",
                        "birthDate": "1960-06-15"
                    }
                },
                {
                    "resource": {
                        "resourceType": "Condition",
                        "id": "cond-1",
                        "code": {
                            "coding": [{"system": "http://hl7.org/fhir/sid/icd-10-cm", "code": "E11", "display": "Type 2 Diabetes"}]
                        },
                        "clinicalStatus": {"coding": [{"code": "active"}]}
                    }
                },
                {
                    "resource": {
                        "resourceType": "MedicationRequest",
                        "id": "mr-1",
                        "status": "active",
                        "intent": "order",
                        "medicationCodeableConcept": {
                            "coding": [{"system": "http://www.nlm.nih.gov/research/umls/rxnorm", "code": "860975", "display": "Metformin 500 MG"}]
                        },
                        "subject": {"reference": "Patient/pt-1"}
                    }
                },
                {
                    "resource": {
                        "resourceType": "Observation",
                        "id": "obs-1",
                        "code": {"coding": [{"system": "http://loinc.org", "code": "4548-4", "display": "HbA1c"}]},
                        "valueQuantity": {"value": 7.5, "unit": "%"},
                        "status": "final"
                    }
                }
            ]
        }"#;

        let bundle = FhirBundle::from_json(json).unwrap();
        let profile = bundle.to_patient_profile().unwrap();
        assert_eq!(profile.sex, "Female");
        assert_eq!(profile.conditions.len(), 1);
        assert_eq!(profile.current_medications.len(), 1);
        assert_eq!(profile.lab_values.len(), 1);
    }
}
