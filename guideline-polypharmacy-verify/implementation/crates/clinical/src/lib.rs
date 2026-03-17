//! # GuardPharma Clinical Module
//!
//! Clinical domain types: conditions, lab values, patient profiles,
//! predicates, and treatment state used throughout the verification pipeline.

use serde::{Deserialize, Serialize};
pub use guardpharma_types::{DrugId, PatientId};

pub mod patient;
pub mod condition;
pub mod medication;
pub mod lab_values;
pub mod state_space;
pub mod fhir;
pub mod temporal;

// ═══════════════════════════════════════════════════════════════════════════
// 1. ClinicalCondition
// ═══════════════════════════════════════════════════════════════════════════

/// Enumerated clinical conditions with ICD-10 mappings.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ClinicalCondition {
    Type2Diabetes,
    Hypertension,
    AtrialFibrillation,
    HeartFailure,
    COPD,
    Depression,
    ChronicPain,
    CKD,
    Asthma,
    Osteoporosis,
    Hypothyroidism,
    GERD,
    Dyslipidemia,
    Obesity,
    Insomnia,
    Anxiety,
    Other(String),
}

impl ClinicalCondition {
    /// Returns the primary ICD-10 code for this condition.
    pub fn icd10_code(&self) -> &str {
        match self {
            Self::Type2Diabetes => "E11",
            Self::Hypertension => "I10",
            Self::AtrialFibrillation => "I48",
            Self::HeartFailure => "I50",
            Self::COPD => "J44",
            Self::Depression => "F33",
            Self::ChronicPain => "G89",
            Self::CKD => "N18",
            Self::Asthma => "J45",
            Self::Osteoporosis => "M81",
            Self::Hypothyroidism => "E03",
            Self::GERD => "K21",
            Self::Dyslipidemia => "E78",
            Self::Obesity => "E66",
            Self::Insomnia => "G47.0",
            Self::Anxiety => "F41",
            Self::Other(_) => "R69",
        }
    }

    /// Returns a human-readable display name.
    pub fn display_name(&self) -> &str {
        match self {
            Self::Type2Diabetes => "Type 2 Diabetes Mellitus",
            Self::Hypertension => "Hypertension",
            Self::AtrialFibrillation => "Atrial Fibrillation",
            Self::HeartFailure => "Heart Failure",
            Self::COPD => "Chronic Obstructive Pulmonary Disease",
            Self::Depression => "Major Depressive Disorder",
            Self::ChronicPain => "Chronic Pain Syndrome",
            Self::CKD => "Chronic Kidney Disease",
            Self::Asthma => "Asthma",
            Self::Osteoporosis => "Osteoporosis",
            Self::Hypothyroidism => "Hypothyroidism",
            Self::GERD => "Gastroesophageal Reflux Disease",
            Self::Dyslipidemia => "Dyslipidemia",
            Self::Obesity => "Obesity",
            Self::Insomnia => "Insomnia",
            Self::Anxiety => "Generalized Anxiety Disorder",
            Self::Other(name) => name.as_str(),
        }
    }
}

impl std::fmt::Display for ClinicalCondition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.display_name())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. ConditionSeverity
// ═══════════════════════════════════════════════════════════════════════════

/// Severity classification for a clinical condition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub enum ConditionSeverity {
    Mild,
    Moderate,
    Severe,
    Critical,
}

impl Default for ConditionSeverity {
    fn default() -> Self {
        Self::Moderate
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. PatientCondition
// ═══════════════════════════════════════════════════════════════════════════

/// A condition present on a patient record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatientCondition {
    pub condition: ClinicalCondition,
    pub severity: ConditionSeverity,
    pub diagnosed_date: Option<String>,
    pub active: bool,
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. LabTest
// ═══════════════════════════════════════════════════════════════════════════

/// Laboratory test identifiers with associated metadata.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LabTest {
    HbA1c,
    SerumCreatinine,
    #[serde(rename = "eGFR")]
    EGFR,
    INR,
    Potassium,
    Sodium,
    ALT,
    AST,
    TSH,
    LDL,
    HDL,
    TotalCholesterol,
    Triglycerides,
    BNP,
    Hemoglobin,
    WBC,
    Platelets,
    BloodPressureSystolic,
    BloodPressureDiastolic,
    HeartRate,
    Other(String),
}

impl LabTest {
    /// Standard unit of measure for this lab test.
    pub fn unit(&self) -> &str {
        match self {
            Self::HbA1c => "%",
            Self::SerumCreatinine => "mg/dL",
            Self::EGFR => "mL/min/1.73m²",
            Self::INR => "ratio",
            Self::Potassium => "mEq/L",
            Self::Sodium => "mEq/L",
            Self::ALT => "U/L",
            Self::AST => "U/L",
            Self::TSH => "mIU/L",
            Self::LDL => "mg/dL",
            Self::HDL => "mg/dL",
            Self::TotalCholesterol => "mg/dL",
            Self::Triglycerides => "mg/dL",
            Self::BNP => "pg/mL",
            Self::Hemoglobin => "g/dL",
            Self::WBC => "×10³/µL",
            Self::Platelets => "×10³/µL",
            Self::BloodPressureSystolic => "mmHg",
            Self::BloodPressureDiastolic => "mmHg",
            Self::HeartRate => "bpm",
            Self::Other(_) => "unknown",
        }
    }

    /// Normal reference range `(low, high)` for this lab test.
    pub fn normal_range(&self) -> (f64, f64) {
        match self {
            Self::HbA1c => (4.0, 5.6),
            Self::SerumCreatinine => (0.7, 1.3),
            Self::EGFR => (90.0, 120.0),
            Self::INR => (0.8, 1.2),
            Self::Potassium => (3.5, 5.0),
            Self::Sodium => (136.0, 145.0),
            Self::ALT => (7.0, 56.0),
            Self::AST => (10.0, 40.0),
            Self::TSH => (0.4, 4.0),
            Self::LDL => (0.0, 100.0),
            Self::HDL => (40.0, 60.0),
            Self::TotalCholesterol => (0.0, 200.0),
            Self::Triglycerides => (0.0, 150.0),
            Self::BNP => (0.0, 100.0),
            Self::Hemoglobin => (12.0, 17.5),
            Self::WBC => (4.5, 11.0),
            Self::Platelets => (150.0, 400.0),
            Self::BloodPressureSystolic => (90.0, 120.0),
            Self::BloodPressureDiastolic => (60.0, 80.0),
            Self::HeartRate => (60.0, 100.0),
            Self::Other(_) => (0.0, f64::MAX),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. LabValue
// ═══════════════════════════════════════════════════════════════════════════

/// A single laboratory measurement with its test type and result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabValue {
    pub test: LabTest,
    pub value: f64,
    pub timestamp: Option<String>,
    pub unit: String,
}

impl LabValue {
    /// Whether the value falls within the normal reference range.
    pub fn is_normal(&self) -> bool {
        let (lo, hi) = self.test.normal_range();
        self.value >= lo && self.value <= hi
    }

    /// Whether the value is at a clinically critical level requiring
    /// immediate attention.
    pub fn is_critical(&self) -> bool {
        match &self.test {
            LabTest::Potassium => self.value < 2.5 || self.value > 6.5,
            LabTest::Sodium => self.value < 120.0 || self.value > 160.0,
            LabTest::INR => self.value > 5.0,
            LabTest::Hemoglobin => self.value < 7.0,
            LabTest::Platelets => self.value < 50.0,
            LabTest::HbA1c => self.value > 14.0,
            LabTest::SerumCreatinine => self.value > 10.0,
            LabTest::EGFR => self.value < 15.0,
            LabTest::ALT => self.value > 1000.0,
            LabTest::AST => self.value > 1000.0,
            LabTest::BNP => self.value > 900.0,
            LabTest::BloodPressureSystolic => self.value > 180.0 || self.value < 70.0,
            LabTest::BloodPressureDiastolic => self.value > 120.0 || self.value < 40.0,
            LabTest::HeartRate => self.value > 150.0 || self.value < 40.0,
            LabTest::WBC => self.value < 2.0 || self.value > 30.0,
            LabTest::TSH => self.value > 50.0 || self.value < 0.01,
            _ => {
                // Generic fallback: critical when value is more than one full
                // range-width beyond either boundary.
                let (lo, hi) = self.test.normal_range();
                let width = hi - lo;
                if width <= 0.0 {
                    return false;
                }
                self.value < lo - width || self.value > hi + width
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. PatientProfile
// ═══════════════════════════════════════════════════════════════════════════

/// Complete patient profile for clinical verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatientProfile {
    pub id: PatientId,
    pub age: u32,
    pub sex: String,
    pub weight_kg: f64,
    pub height_cm: f64,
    pub conditions: Vec<PatientCondition>,
    pub lab_values: Vec<LabValue>,
    pub current_medications: Vec<DrugId>,
    pub allergies: Vec<String>,
    pub renal_function_egfr: Option<f64>,
    pub hepatic_function: Option<String>,
}

impl PatientProfile {
    /// Returns `true` if the patient has an **active** instance of `condition`.
    pub fn has_condition(&self, condition: &ClinicalCondition) -> bool {
        self.conditions
            .iter()
            .any(|pc| pc.active && &pc.condition == condition)
    }

    /// Returns the most recent lab value for the given test (last in the list).
    pub fn latest_lab(&self, test: &LabTest) -> Option<&LabValue> {
        self.lab_values.iter().rev().find(|lv| &lv.test == test)
    }

    /// Whether the patient is currently taking `drug`.
    pub fn is_on_medication(&self, drug: &DrugId) -> bool {
        self.current_medications.iter().any(|d| d == drug)
    }

    /// Body mass index (kg/m²).
    pub fn bmi(&self) -> f64 {
        let height_m = self.height_cm / 100.0;
        if height_m <= 0.0 {
            return 0.0;
        }
        self.weight_kg / (height_m * height_m)
    }

    /// CKD stage (1–5) derived from eGFR, if available.
    pub fn ckd_stage(&self) -> Option<u8> {
        let egfr = self
            .renal_function_egfr
            .or_else(|| self.latest_lab(&LabTest::EGFR).map(|lv| lv.value))?;
        Some(if egfr >= 90.0 {
            1
        } else if egfr >= 60.0 {
            2
        } else if egfr >= 30.0 {
            3
        } else if egfr >= 15.0 {
            4
        } else {
            5
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 7. ClinicalPredicate
// ═══════════════════════════════════════════════════════════════════════════

/// Composable boolean predicates over a [`PatientProfile`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClinicalPredicate {
    HasCondition(ClinicalCondition),
    LabAbove(LabTest, f64),
    LabBelow(LabTest, f64),
    LabInRange(LabTest, f64, f64),
    OnMedication(DrugId),
    AgeAbove(u32),
    AgeBelow(u32),
    BMIAbove(f64),
    And(Box<ClinicalPredicate>, Box<ClinicalPredicate>),
    Or(Box<ClinicalPredicate>, Box<ClinicalPredicate>),
    Not(Box<ClinicalPredicate>),
}

impl ClinicalPredicate {
    /// Evaluate this predicate against a patient profile.
    pub fn evaluate(&self, patient: &PatientProfile) -> bool {
        match self {
            Self::HasCondition(c) => patient.has_condition(c),
            Self::LabAbove(test, threshold) => patient
                .latest_lab(test)
                .map_or(false, |lv| lv.value > *threshold),
            Self::LabBelow(test, threshold) => patient
                .latest_lab(test)
                .map_or(false, |lv| lv.value < *threshold),
            Self::LabInRange(test, lo, hi) => patient
                .latest_lab(test)
                .map_or(false, |lv| lv.value >= *lo && lv.value <= *hi),
            Self::OnMedication(drug) => patient.is_on_medication(drug),
            Self::AgeAbove(age) => patient.age > *age,
            Self::AgeBelow(age) => patient.age < *age,
            Self::BMIAbove(threshold) => patient.bmi() > *threshold,
            Self::And(a, b) => a.evaluate(patient) && b.evaluate(patient),
            Self::Or(a, b) => a.evaluate(patient) || b.evaluate(patient),
            Self::Not(inner) => !inner.evaluate(patient),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 8. ClinicalState
// ═══════════════════════════════════════════════════════════════════════════

/// Snapshot of a patient's clinical state at a point in simulated time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClinicalState {
    pub patient: PatientProfile,
    pub time_hours: f64,
    pub active_treatments: Vec<ActiveTreatment>,
}

impl ClinicalState {
    /// Move the simulation clock forward by `hours`.
    pub fn advance_time(&mut self, hours: f64) {
        self.time_hours += hours;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 9. ActiveTreatment
// ═══════════════════════════════════════════════════════════════════════════

/// A treatment currently being administered, with its dose history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveTreatment {
    pub drug_id: DrugId,
    pub dose_mg: f64,
    pub interval_hours: f64,
    pub start_time: f64,
    pub adjustments: Vec<DoseAdjustment>,
}

// ═══════════════════════════════════════════════════════════════════════════
// 10. DoseAdjustment
// ═══════════════════════════════════════════════════════════════════════════

/// Record of a dose change applied to an active treatment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoseAdjustment {
    pub time: f64,
    pub new_dose_mg: f64,
    pub reason: String,
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a minimal patient profile for testing.
    fn test_patient() -> PatientProfile {
        PatientProfile {
            id: PatientId::new(),
            age: 68,
            sex: "Male".into(),
            weight_kg: 85.0,
            height_cm: 175.0,
            conditions: vec![
                PatientCondition {
                    condition: ClinicalCondition::Type2Diabetes,
                    severity: ConditionSeverity::Moderate,
                    diagnosed_date: Some("2019-03-15".into()),
                    active: true,
                },
                PatientCondition {
                    condition: ClinicalCondition::Hypertension,
                    severity: ConditionSeverity::Mild,
                    diagnosed_date: None,
                    active: true,
                },
                PatientCondition {
                    condition: ClinicalCondition::Asthma,
                    severity: ConditionSeverity::Mild,
                    diagnosed_date: None,
                    active: false,
                },
            ],
            lab_values: vec![
                LabValue {
                    test: LabTest::HbA1c,
                    value: 7.2,
                    timestamp: Some("2024-01-10T08:00:00Z".into()),
                    unit: "%".into(),
                },
                LabValue {
                    test: LabTest::EGFR,
                    value: 55.0,
                    timestamp: Some("2024-01-10T08:00:00Z".into()),
                    unit: "mL/min/1.73m²".into(),
                },
                LabValue {
                    test: LabTest::Potassium,
                    value: 4.2,
                    timestamp: None,
                    unit: "mEq/L".into(),
                },
                LabValue {
                    test: LabTest::INR,
                    value: 2.5,
                    timestamp: None,
                    unit: "ratio".into(),
                },
            ],
            current_medications: vec![
                DrugId::new("metformin"),
                DrugId::new("lisinopril"),
                DrugId::new("warfarin"),
            ],
            allergies: vec!["Penicillin".into()],
            renal_function_egfr: Some(55.0),
            hepatic_function: None,
        }
    }

    // ── ClinicalCondition ────────────────────────────────────────────────

    #[test]
    fn condition_icd10_codes() {
        assert_eq!(ClinicalCondition::Type2Diabetes.icd10_code(), "E11");
        assert_eq!(ClinicalCondition::Hypertension.icd10_code(), "I10");
        assert_eq!(ClinicalCondition::COPD.icd10_code(), "J44");
        assert_eq!(
            ClinicalCondition::Other("Lupus".into()).icd10_code(),
            "R69"
        );
    }

    #[test]
    fn condition_display_names() {
        assert_eq!(
            ClinicalCondition::AtrialFibrillation.display_name(),
            "Atrial Fibrillation"
        );
        assert_eq!(
            ClinicalCondition::GERD.display_name(),
            "Gastroesophageal Reflux Disease"
        );
        assert_eq!(
            ClinicalCondition::Other("Lupus".into()).display_name(),
            "Lupus"
        );
    }

    // ── LabTest ──────────────────────────────────────────────────────────

    #[test]
    fn lab_test_units_and_ranges() {
        assert_eq!(LabTest::HbA1c.unit(), "%");
        assert_eq!(LabTest::Potassium.unit(), "mEq/L");
        assert_eq!(LabTest::HeartRate.unit(), "bpm");

        let (lo, hi) = LabTest::Sodium.normal_range();
        assert!((lo - 136.0).abs() < f64::EPSILON);
        assert!((hi - 145.0).abs() < f64::EPSILON);
    }

    // ── LabValue ─────────────────────────────────────────────────────────

    #[test]
    fn lab_value_is_normal() {
        let normal_k = LabValue {
            test: LabTest::Potassium,
            value: 4.0,
            timestamp: None,
            unit: "mEq/L".into(),
        };
        assert!(normal_k.is_normal());

        let high_k = LabValue {
            test: LabTest::Potassium,
            value: 5.5,
            timestamp: None,
            unit: "mEq/L".into(),
        };
        assert!(!high_k.is_normal());
    }

    #[test]
    fn lab_value_is_critical() {
        let critical_k = LabValue {
            test: LabTest::Potassium,
            value: 7.0,
            timestamp: None,
            unit: "mEq/L".into(),
        };
        assert!(critical_k.is_critical());

        let normal_k = LabValue {
            test: LabTest::Potassium,
            value: 4.0,
            timestamp: None,
            unit: "mEq/L".into(),
        };
        assert!(!normal_k.is_critical());

        let critical_inr = LabValue {
            test: LabTest::INR,
            value: 6.0,
            timestamp: None,
            unit: "ratio".into(),
        };
        assert!(critical_inr.is_critical());
    }

    // ── PatientProfile ───────────────────────────────────────────────────

    #[test]
    fn patient_has_condition() {
        let p = test_patient();
        assert!(p.has_condition(&ClinicalCondition::Type2Diabetes));
        assert!(p.has_condition(&ClinicalCondition::Hypertension));
        // Asthma is inactive, so should be false
        assert!(!p.has_condition(&ClinicalCondition::Asthma));
        assert!(!p.has_condition(&ClinicalCondition::Depression));
    }

    #[test]
    fn patient_bmi() {
        let p = test_patient();
        let expected = 85.0 / (1.75 * 1.75);
        assert!((p.bmi() - expected).abs() < 0.01);
    }

    #[test]
    fn patient_ckd_stage() {
        let p = test_patient();
        // eGFR = 55 → stage 3
        assert_eq!(p.ckd_stage(), Some(3));

        let mut p2 = test_patient();
        p2.renal_function_egfr = Some(95.0);
        assert_eq!(p2.ckd_stage(), Some(1));

        let mut p3 = test_patient();
        p3.renal_function_egfr = Some(10.0);
        assert_eq!(p3.ckd_stage(), Some(5));
    }

    #[test]
    fn patient_latest_lab() {
        let p = test_patient();
        let hba1c = p.latest_lab(&LabTest::HbA1c).unwrap();
        assert!((hba1c.value - 7.2).abs() < f64::EPSILON);

        assert!(p.latest_lab(&LabTest::BNP).is_none());
    }

    #[test]
    fn patient_is_on_medication() {
        let p = test_patient();
        assert!(p.is_on_medication(&DrugId::new("warfarin")));
        assert!(!p.is_on_medication(&DrugId::new("aspirin")));
    }

    // ── ClinicalPredicate ────────────────────────────────────────────────

    #[test]
    fn predicate_evaluate_simple() {
        let p = test_patient();

        let has_dm = ClinicalPredicate::HasCondition(ClinicalCondition::Type2Diabetes);
        assert!(has_dm.evaluate(&p));

        let age_over_65 = ClinicalPredicate::AgeAbove(65);
        assert!(age_over_65.evaluate(&p));

        let age_over_70 = ClinicalPredicate::AgeAbove(70);
        assert!(!age_over_70.evaluate(&p));

        let on_warfarin = ClinicalPredicate::OnMedication(DrugId::new("warfarin"));
        assert!(on_warfarin.evaluate(&p));

        let hba1c_above_7 = ClinicalPredicate::LabAbove(LabTest::HbA1c, 7.0);
        assert!(hba1c_above_7.evaluate(&p));

        let hba1c_below_6 = ClinicalPredicate::LabBelow(LabTest::HbA1c, 6.0);
        assert!(!hba1c_below_6.evaluate(&p));
    }

    #[test]
    fn predicate_evaluate_compound() {
        let p = test_patient();

        let dm_and_elderly = ClinicalPredicate::And(
            Box::new(ClinicalPredicate::HasCondition(
                ClinicalCondition::Type2Diabetes,
            )),
            Box::new(ClinicalPredicate::AgeAbove(60)),
        );
        assert!(dm_and_elderly.evaluate(&p));

        let not_on_aspirin =
            ClinicalPredicate::Not(Box::new(ClinicalPredicate::OnMedication(DrugId::new(
                "aspirin",
            ))));
        assert!(not_on_aspirin.evaluate(&p));

        let depression_or_anxiety = ClinicalPredicate::Or(
            Box::new(ClinicalPredicate::HasCondition(
                ClinicalCondition::Depression,
            )),
            Box::new(ClinicalPredicate::HasCondition(
                ClinicalCondition::Anxiety,
            )),
        );
        assert!(!depression_or_anxiety.evaluate(&p));
    }

    #[test]
    fn predicate_lab_in_range() {
        let p = test_patient();
        let k_in_range = ClinicalPredicate::LabInRange(LabTest::Potassium, 3.5, 5.0);
        assert!(k_in_range.evaluate(&p));

        let k_tight = ClinicalPredicate::LabInRange(LabTest::Potassium, 4.5, 5.0);
        assert!(!k_tight.evaluate(&p));
    }

    #[test]
    fn predicate_bmi_above() {
        let p = test_patient();
        // BMI ≈ 27.76
        let obese = ClinicalPredicate::BMIAbove(30.0);
        assert!(!obese.evaluate(&p));

        let overweight = ClinicalPredicate::BMIAbove(25.0);
        assert!(overweight.evaluate(&p));
    }

    // ── ClinicalState ────────────────────────────────────────────────────

    #[test]
    fn clinical_state_advance_time() {
        let p = test_patient();
        let mut state = ClinicalState {
            patient: p,
            time_hours: 0.0,
            active_treatments: vec![ActiveTreatment {
                drug_id: DrugId::new("warfarin"),
                dose_mg: 5.0,
                interval_hours: 24.0,
                start_time: 0.0,
                adjustments: vec![],
            }],
        };
        state.advance_time(12.0);
        assert!((state.time_hours - 12.0).abs() < f64::EPSILON);
        state.advance_time(6.0);
        assert!((state.time_hours - 18.0).abs() < f64::EPSILON);
    }

    // ── Serialization round-trip ─────────────────────────────────────────

    #[test]
    fn serde_roundtrip_patient_profile() {
        let p = test_patient();
        let json = serde_json::to_string(&p).expect("serialize");
        let p2: PatientProfile = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(p.age, p2.age);
        assert_eq!(p.current_medications.len(), p2.current_medications.len());
    }

    #[test]
    fn serde_roundtrip_predicate() {
        let pred = ClinicalPredicate::And(
            Box::new(ClinicalPredicate::HasCondition(
                ClinicalCondition::Type2Diabetes,
            )),
            Box::new(ClinicalPredicate::Not(Box::new(
                ClinicalPredicate::AgeBelow(18),
            ))),
        );
        let json = serde_json::to_string(&pred).expect("serialize");
        let pred2: ClinicalPredicate = serde_json::from_str(&json).expect("deserialize");
        let p = test_patient();
        assert_eq!(pred.evaluate(&p), pred2.evaluate(&p));
    }
}
