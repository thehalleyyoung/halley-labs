//! FHIR R4 Patient resource parsing → internal Patient / PatientProfile.

use serde::{Deserialize, Serialize};

use guardpharma_clinical::{
    PatientProfile, PatientCondition, ClinicalCondition, ConditionSeverity,
    LabValue, LabTest, PatientId,
};

use super::types::{CodeableConcept, HumanName, Identifier, Meta, Reference};
use crate::error::FhirInteropError;

/// FHIR R4 Patient resource (subset of fields relevant to GuardPharma).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FhirPatientResource {
    #[serde(rename = "resourceType")]
    pub resource_type: String,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub meta: Option<Meta>,
    #[serde(default)]
    pub identifier: Vec<Identifier>,
    #[serde(default)]
    pub name: Vec<HumanName>,
    #[serde(default)]
    pub gender: Option<String>,
    #[serde(rename = "birthDate", default)]
    pub birth_date: Option<String>,
    #[serde(default)]
    pub active: Option<bool>,
}

impl FhirPatientResource {
    /// Parse from a JSON string.
    pub fn from_json(json: &str) -> Result<Self, FhirInteropError> {
        let res: Self = serde_json::from_str(json)?;
        if res.resource_type != "Patient" {
            return Err(FhirInteropError::ResourceTypeMismatch {
                expected: "Patient".into(),
                got: res.resource_type,
            });
        }
        Ok(res)
    }

    /// Convert to the internal PatientProfile.
    ///
    /// Produces a minimal profile; conditions, labs, and medications should be
    /// populated separately from their own FHIR resources.
    pub fn to_patient_profile(&self) -> Result<PatientProfile, FhirInteropError> {
        let id = self
            .id
            .as_deref()
            .map(|mrn| {
                let u = uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_DNS, mrn.as_bytes());
                PatientId::from_uuid(u)
            })
            .unwrap_or_else(PatientId::new);

        let age = estimate_age(self.birth_date.as_deref()).unwrap_or(0);

        let sex = self
            .gender
            .as_deref()
            .map(|g| match g {
                "male" => "Male".to_string(),
                "female" => "Female".to_string(),
                other => other.to_string(),
            })
            .unwrap_or_else(|| "Unknown".to_string());

        Ok(PatientProfile {
            id,
            age,
            sex,
            weight_kg: 70.0,  // default – not in Patient resource
            height_cm: 170.0, // default
            conditions: Vec::new(),
            lab_values: Vec::new(),
            current_medications: Vec::new(),
            allergies: Vec::new(),
            renal_function_egfr: None,
            hepatic_function: None,
        })
    }
}

/// FHIR R4 Condition resource.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FhirConditionResource {
    #[serde(rename = "resourceType")]
    pub resource_type: String,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub code: Option<CodeableConcept>,
    #[serde(rename = "clinicalStatus", default)]
    pub clinical_status: Option<CodeableConcept>,
    #[serde(default)]
    pub subject: Option<Reference>,
    #[serde(rename = "onsetDateTime", default)]
    pub onset_date_time: Option<String>,
    #[serde(default)]
    pub severity: Option<CodeableConcept>,
}

impl FhirConditionResource {
    /// Parse from JSON.
    pub fn from_json(json: &str) -> Result<Self, FhirInteropError> {
        let res: Self = serde_json::from_str(json)?;
        if res.resource_type != "Condition" {
            return Err(FhirInteropError::ResourceTypeMismatch {
                expected: "Condition".into(),
                got: res.resource_type,
            });
        }
        Ok(res)
    }

    /// Convert to the internal PatientCondition.
    pub fn to_patient_condition(&self) -> PatientCondition {
        let condition = self
            .code
            .as_ref()
            .and_then(|cc| cc.icd10_code())
            .map(icd10_to_clinical_condition)
            .unwrap_or_else(|| {
                let name = self
                    .code
                    .as_ref()
                    .and_then(|cc| cc.display_text())
                    .unwrap_or("Unknown")
                    .to_string();
                ClinicalCondition::Other(name)
            });

        let active = self
            .clinical_status
            .as_ref()
            .and_then(|cs| cs.coding.first())
            .and_then(|c| c.code.as_deref())
            .map(|s| s == "active" || s == "recurrence" || s == "relapse")
            .unwrap_or(true);

        let severity = self
            .severity
            .as_ref()
            .and_then(|s| s.display_text())
            .map(text_to_severity)
            .unwrap_or_default();

        PatientCondition {
            condition,
            severity,
            diagnosed_date: self.onset_date_time.clone(),
            active,
        }
    }
}

/// FHIR R4 Observation resource (laboratory results).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FhirObservationResource {
    #[serde(rename = "resourceType")]
    pub resource_type: String,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub code: Option<CodeableConcept>,
    #[serde(rename = "valueQuantity", default)]
    pub value_quantity: Option<super::types::Quantity>,
    #[serde(rename = "effectiveDateTime", default)]
    pub effective_date_time: Option<String>,
    #[serde(default)]
    pub subject: Option<Reference>,
    #[serde(default)]
    pub status: Option<String>,
}

impl FhirObservationResource {
    /// Parse from JSON.
    pub fn from_json(json: &str) -> Result<Self, FhirInteropError> {
        let res: Self = serde_json::from_str(json)?;
        if res.resource_type != "Observation" {
            return Err(FhirInteropError::ResourceTypeMismatch {
                expected: "Observation".into(),
                got: res.resource_type,
            });
        }
        Ok(res)
    }

    /// Convert to the internal LabValue, if a numeric value is present.
    pub fn to_lab_value(&self) -> Option<LabValue> {
        let qty = self.value_quantity.as_ref()?;
        let value = qty.value?;

        let test = self
            .code
            .as_ref()
            .and_then(|cc| cc.loinc_code())
            .map(loinc_to_lab_test)
            .unwrap_or_else(|| {
                let name = self
                    .code
                    .as_ref()
                    .and_then(|cc| cc.display_text())
                    .unwrap_or("Unknown")
                    .to_string();
                LabTest::Other(name)
            });

        let unit = qty
            .unit
            .clone()
            .unwrap_or_else(|| test.unit().to_string());

        Some(LabValue {
            test,
            value,
            timestamp: self.effective_date_time.clone(),
            unit,
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Mapping helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Map an ICD-10 code prefix to a ClinicalCondition variant.
fn icd10_to_clinical_condition(code: &str) -> ClinicalCondition {
    let prefix = code.split('.').next().unwrap_or(code);
    match prefix {
        "E11" => ClinicalCondition::Type2Diabetes,
        "I10" => ClinicalCondition::Hypertension,
        "I48" => ClinicalCondition::AtrialFibrillation,
        "I50" => ClinicalCondition::HeartFailure,
        "J44" => ClinicalCondition::COPD,
        "F33" | "F32" => ClinicalCondition::Depression,
        "G89" | "M54" => ClinicalCondition::ChronicPain,
        "N18" => ClinicalCondition::CKD,
        "J45" => ClinicalCondition::Asthma,
        "M81" | "M80" => ClinicalCondition::Osteoporosis,
        "E03" => ClinicalCondition::Hypothyroidism,
        "K21" => ClinicalCondition::GERD,
        "E78" => ClinicalCondition::Dyslipidemia,
        "E66" => ClinicalCondition::Obesity,
        "G47" => ClinicalCondition::Insomnia,
        "F41" => ClinicalCondition::Anxiety,
        _ => ClinicalCondition::Other(code.to_string()),
    }
}

/// Map a LOINC code to a LabTest variant.
fn loinc_to_lab_test(code: &str) -> LabTest {
    match code {
        "4548-4" | "17856-6" => LabTest::HbA1c,
        "2160-0" => LabTest::SerumCreatinine,
        "33914-3" | "62238-1" | "48642-3" | "48643-1" => LabTest::EGFR,
        "6301-6" | "34714-6" => LabTest::INR,
        "2823-3" | "6298-4" => LabTest::Potassium,
        "2951-2" | "2947-0" => LabTest::Sodium,
        "1742-6" | "1744-2" => LabTest::ALT,
        "1920-8" | "30239-8" => LabTest::AST,
        "3016-3" | "11580-8" => LabTest::TSH,
        "2089-1" | "18262-6" => LabTest::LDL,
        "2085-9" => LabTest::HDL,
        "2093-3" => LabTest::TotalCholesterol,
        "2571-8" => LabTest::Triglycerides,
        "30934-4" | "33762-6" => LabTest::BNP,
        "718-7" | "20509-6" => LabTest::Hemoglobin,
        "6690-2" | "26464-8" => LabTest::WBC,
        "777-3" | "26515-7" => LabTest::Platelets,
        "8480-6" => LabTest::BloodPressureSystolic,
        "8462-4" => LabTest::BloodPressureDiastolic,
        "8867-4" => LabTest::HeartRate,
        _ => LabTest::Other(format!("LOINC:{code}")),
    }
}

/// Map severity display text to a ConditionSeverity.
fn text_to_severity(text: &str) -> ConditionSeverity {
    match text.to_lowercase().as_str() {
        "mild" => ConditionSeverity::Mild,
        "moderate" => ConditionSeverity::Moderate,
        "severe" => ConditionSeverity::Severe,
        _ => ConditionSeverity::Moderate,
    }
}

/// Rough age estimate from a birth-date string (YYYY or YYYY-MM-DD).
fn estimate_age(birth_date: Option<&str>) -> Option<u32> {
    let bd = birth_date?;
    let year: i32 = bd.get(..4)?.parse().ok()?;
    let now_year = chrono::Utc::now().year();
    let age = now_year - year;
    if age >= 0 { Some(age as u32) } else { None }
}

use chrono::Datelike;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_fhir_patient() {
        let json = r#"{
            "resourceType": "Patient",
            "id": "example-123",
            "gender": "male",
            "birthDate": "1956-01-01",
            "name": [{"family": "Smith", "given": ["John"]}]
        }"#;
        let p = FhirPatientResource::from_json(json).unwrap();
        assert_eq!(p.gender.as_deref(), Some("male"));
        let profile = p.to_patient_profile().unwrap();
        assert_eq!(profile.sex, "Male");
        assert!(profile.age > 60);
    }

    #[test]
    fn parse_fhir_condition() {
        let json = r#"{
            "resourceType": "Condition",
            "id": "cond-1",
            "code": {
                "coding": [{"system": "http://hl7.org/fhir/sid/icd-10-cm", "code": "I10", "display": "Hypertension"}]
            },
            "clinicalStatus": {
                "coding": [{"code": "active"}]
            }
        }"#;
        let c = FhirConditionResource::from_json(json).unwrap();
        let pc = c.to_patient_condition();
        assert_eq!(pc.condition, ClinicalCondition::Hypertension);
        assert!(pc.active);
    }

    #[test]
    fn parse_fhir_observation() {
        let json = r#"{
            "resourceType": "Observation",
            "id": "obs-1",
            "code": {
                "coding": [{"system": "http://loinc.org", "code": "4548-4", "display": "HbA1c"}]
            },
            "valueQuantity": {"value": 7.2, "unit": "%"}
        }"#;
        let o = FhirObservationResource::from_json(json).unwrap();
        let lv = o.to_lab_value().unwrap();
        assert_eq!(lv.test, LabTest::HbA1c);
        assert!((lv.value - 7.2).abs() < f64::EPSILON);
    }
}
