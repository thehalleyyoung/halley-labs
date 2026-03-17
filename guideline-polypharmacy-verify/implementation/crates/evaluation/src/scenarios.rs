//! Test scenario generation for GuardPharma evaluation.
//!
//! Provides common clinical scenarios (elderly multimorbid, cardiac, renal, etc.),
//! a random scenario generator, and adversarial stress-test cases.

use serde::{Deserialize, Serialize};
use std::fmt;

use guardpharma_types::{
    AscitesGrade, ChildPughClass, DosingSchedule, DrugId, DrugInfo, DrugRoute,
    GuidelineId, PatientId, PatientInfo, RenalFunction, Sex, Severity,
};
use guardpharma_clinical::{
    ActiveMedication, ClinicalCondition, ConditionSeverity, GuidelineReference,
    LabValue, PatientProfile,
};

use crate::benchmark::BenchmarkSetup;

// ═══════════════════════════════════════════════════════════════════════════
// Scenario Severity
// ═══════════════════════════════════════════════════════════════════════════

/// Difficulty level of a generated scenario.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ScenarioSeverity {
    /// Few drugs, no interactions.
    Easy,
    /// Moderate polypharmacy, some known interactions.
    Medium,
    /// Many drugs with several interactions.
    Hard,
    /// Extreme polypharmacy with cascading interactions — stress-test.
    Adversarial,
}

impl fmt::Display for ScenarioSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Easy => write!(f, "Easy"),
            Self::Medium => write!(f, "Medium"),
            Self::Hard => write!(f, "Hard"),
            Self::Adversarial => write!(f, "Adversarial"),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Helper: medication builder
// ═══════════════════════════════════════════════════════════════════════════

fn med(name: &str, class: &str, dose: f64, interval: f64) -> ActiveMedication {
    ActiveMedication::new(
        DrugId::new(name),
        DrugInfo::new(name, class),
        DosingSchedule::new(dose, interval),
    )
}

fn med_iv(name: &str, class: &str, dose: f64, interval: f64) -> ActiveMedication {
    let mut m = med(name, class, dose, interval);
    m.dosing.route = DrugRoute::Intravenous;
    m
}

fn condition(name: &str, icd10: &str, severity: ConditionSeverity) -> ClinicalCondition {
    ClinicalCondition {
        name: name.to_string(),
        icd10_code: Some(icd10.to_string()),
        severity,
        active: true,
    }
}

fn lab(name: &str, value: f64, unit: &str, lo: f64, hi: f64) -> (String, LabValue) {
    let mut lv = LabValue::new(name, value, unit);
    lv.reference_low = Some(lo);
    lv.reference_high = Some(hi);
    (name.to_string(), lv)
}

fn make_guideline(name: &str, meds: Vec<ActiveMedication>) -> GuidelineReference {
    let mut gl = GuidelineReference::new(name);
    for m in meds {
        gl.recommended_medications.push(m);
    }
    gl
}

// ═══════════════════════════════════════════════════════════════════════════
// ScenarioGenerator
// ═══════════════════════════════════════════════════════════════════════════

/// Main entry-point for scenario generation.
#[derive(Debug, Clone)]
pub struct ScenarioGenerator;

impl ScenarioGenerator {
    /// Generate a polypharmacy scenario with the given number of drugs.
    pub fn generate_polypharmacy_scenario(n_drugs: usize) -> BenchmarkSetup {
        let drug_pool = Self::drug_pool();
        let n = n_drugs.min(drug_pool.len());

        let mut profile = PatientProfile::new(PatientId::new(), PatientInfo {
            age_years: 72.0,
            weight_kg: 75.0,
            height_cm: 170.0,
            sex: Sex::Male,
            serum_creatinine: 1.2,
            ..PatientInfo::default()
        });

        for (name, class, dose, interval) in drug_pool.into_iter().take(n) {
            profile.add_medication(med(name, class, dose, interval));
        }

        let n_gl = (n / 3).max(1);
        let guidelines: Vec<GuidelineReference> = (0..n_gl)
            .map(|i| GuidelineReference::new(&format!("GL-poly-{}", i)))
            .collect();

        BenchmarkSetup::new(profile)
            .with_guidelines(guidelines)
            .with_time_horizon(168.0)
    }

    /// Standard drug pool (name, class, dose_mg, interval_hours).
    fn drug_pool() -> Vec<(&'static str, &'static str, f64, f64)> {
        vec![
            ("warfarin", "Anticoagulant", 5.0, 24.0),
            ("aspirin", "NSAID", 81.0, 24.0),
            ("metformin", "Antidiabetic", 500.0, 12.0),
            ("lisinopril", "ACE Inhibitor", 10.0, 24.0),
            ("amlodipine", "CCB", 5.0, 24.0),
            ("atorvastatin", "Statin", 20.0, 24.0),
            ("metoprolol", "Beta Blocker", 50.0, 12.0),
            ("omeprazole", "PPI", 20.0, 24.0),
            ("furosemide", "Loop Diuretic", 40.0, 24.0),
            ("spironolactone", "K-sparing Diuretic", 25.0, 24.0),
            ("amiodarone", "Antiarrhythmic", 200.0, 24.0),
            ("digoxin", "Cardiac Glycoside", 0.125, 24.0),
            ("simvastatin", "Statin", 20.0, 24.0),
            ("fluconazole", "Antifungal", 100.0, 24.0),
            ("ciprofloxacin", "Fluoroquinolone", 500.0, 12.0),
            ("fluoxetine", "SSRI", 20.0, 24.0),
            ("tramadol", "Opioid", 50.0, 6.0),
            ("verapamil", "CCB", 80.0, 8.0),
            ("clopidogrel", "Antiplatelet", 75.0, 24.0),
            ("levothyroxine", "Thyroid Hormone", 0.1, 24.0),
            ("sertraline", "SSRI", 50.0, 24.0),
            ("ibuprofen", "NSAID", 400.0, 8.0),
            ("cyclosporine", "Immunosuppressant", 100.0, 12.0),
            ("tacrolimus", "Immunosuppressant", 2.0, 12.0),
            ("erythromycin", "Macrolide", 250.0, 6.0),
            ("clarithromycin", "Macrolide", 500.0, 12.0),
            ("diltiazem", "CCB", 120.0, 8.0),
            ("haloperidol", "Antipsychotic", 5.0, 12.0),
            ("morphine", "Opioid", 10.0, 4.0),
            ("oxycodone", "Opioid", 5.0, 6.0),
        ]
    }

    /// Generate a scenario at the given severity level.
    pub fn generate_by_severity(severity: ScenarioSeverity) -> BenchmarkSetup {
        match severity {
            ScenarioSeverity::Easy => Self::generate_polypharmacy_scenario(3),
            ScenarioSeverity::Medium => Self::generate_polypharmacy_scenario(8),
            ScenarioSeverity::Hard => Self::generate_polypharmacy_scenario(15),
            ScenarioSeverity::Adversarial => generate_adversarial_scenario(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CommonScenarios
// ═══════════════════════════════════════════════════════════════════════════

/// Static methods returning clinically realistic benchmark setups.
pub struct CommonScenarios;

impl CommonScenarios {
    /// 75-year-old with T2DM + HTN + AF + HF + CKD (typical Medicare patient).
    pub fn elderly_multimorbid() -> BenchmarkSetup {
        let mut profile = PatientProfile::new(PatientId::new(), PatientInfo {
            age_years: 75.0,
            weight_kg: 82.0,
            height_cm: 170.0,
            sex: Sex::Male,
            serum_creatinine: 1.8,
            albumin: Some(3.5),
            bilirubin: Some(0.9),
            inr: Some(2.3),
            encephalopathy_grade: Some(0),
            ascites: Some(AscitesGrade::None),
        });

        // Medications (12 drugs — typical Medicare polypharmacy).
        profile.add_medication(med("warfarin", "Anticoagulant", 5.0, 24.0));
        profile.add_medication(med("metformin", "Antidiabetic", 500.0, 12.0));
        profile.add_medication(med("lisinopril", "ACE Inhibitor", 10.0, 24.0));
        profile.add_medication(med("amlodipine", "CCB", 5.0, 24.0));
        profile.add_medication(med("atorvastatin", "Statin", 40.0, 24.0));
        profile.add_medication(med("metoprolol", "Beta Blocker", 50.0, 12.0));
        profile.add_medication(med("furosemide", "Loop Diuretic", 40.0, 24.0));
        profile.add_medication(med("spironolactone", "K-sparing Diuretic", 25.0, 24.0));
        profile.add_medication(med("digoxin", "Cardiac Glycoside", 0.125, 24.0));
        profile.add_medication(med("omeprazole", "PPI", 20.0, 24.0));
        profile.add_medication(med("aspirin", "NSAID", 81.0, 24.0));
        profile.add_medication(med("levothyroxine", "Thyroid Hormone", 0.075, 24.0));

        // Conditions.
        profile.add_condition(condition("Type 2 Diabetes Mellitus", "E11.65", ConditionSeverity::Moderate));
        profile.add_condition(condition("Essential Hypertension", "I10", ConditionSeverity::Moderate));
        profile.add_condition(condition("Atrial Fibrillation", "I48.91", ConditionSeverity::Moderate));
        profile.add_condition(condition("Heart Failure", "I50.9", ConditionSeverity::Severe));
        profile.add_condition(condition("Chronic Kidney Disease Stage 3", "N18.3", ConditionSeverity::Moderate));
        profile.add_condition(condition("Hypothyroidism", "E03.9", ConditionSeverity::Mild));

        // Lab values.
        let labs = vec![
            lab("HbA1c", 7.2, "%", 4.0, 5.6),
            lab("eGFR", 42.0, "mL/min/1.73m2", 60.0, 120.0),
            lab("INR", 2.3, "ratio", 0.9, 1.1),
            lab("Potassium", 4.8, "mEq/L", 3.5, 5.0),
            lab("BNP", 450.0, "pg/mL", 0.0, 100.0),
            lab("TSH", 3.5, "mIU/L", 0.4, 4.0),
            lab("Creatinine", 1.8, "mg/dL", 0.7, 1.3),
        ];
        for (name, lv) in labs {
            profile.add_lab_value(&name, lv);
        }
        profile.renal_function = RenalFunction::Moderate;

        let guidelines = vec![
            make_guideline("AHA/ACC AF Guidelines 2023", vec![
                med("warfarin", "Anticoagulant", 5.0, 24.0),
                med("metoprolol", "Beta Blocker", 50.0, 12.0),
            ]),
            make_guideline("KDIGO CKD Guidelines", vec![
                med("lisinopril", "ACE Inhibitor", 10.0, 24.0),
            ]),
            make_guideline("ADA T2DM Guidelines", vec![
                med("metformin", "Antidiabetic", 500.0, 12.0),
            ]),
            make_guideline("AHA/ACC HF Guidelines", vec![
                med("metoprolol", "Beta Blocker", 50.0, 12.0),
                med("lisinopril", "ACE Inhibitor", 10.0, 24.0),
                med("furosemide", "Loop Diuretic", 40.0, 24.0),
                med("spironolactone", "K-sparing Diuretic", 25.0, 24.0),
            ]),
        ];

        BenchmarkSetup::new(profile)
            .with_guidelines(guidelines)
            .with_properties(vec![
                "no_therapeutic_window_violation".into(),
                "no_critical_interaction".into(),
                "renal_dose_adjustment".into(),
                "no_qt_prolongation_combo".into(),
            ])
            .with_time_horizon(168.0)
    }

    /// T2DM + HTN only — simpler scenario.
    pub fn diabetes_hypertension() -> BenchmarkSetup {
        let mut profile = PatientProfile::new(PatientId::new(), PatientInfo {
            age_years: 58.0,
            weight_kg: 92.0,
            height_cm: 175.0,
            sex: Sex::Female,
            serum_creatinine: 0.9,
            ..PatientInfo::default()
        });

        profile.add_medication(med("metformin", "Antidiabetic", 1000.0, 12.0));
        profile.add_medication(med("glipizide", "Sulfonylurea", 5.0, 24.0));
        profile.add_medication(med("lisinopril", "ACE Inhibitor", 20.0, 24.0));
        profile.add_medication(med("amlodipine", "CCB", 10.0, 24.0));
        profile.add_medication(med("atorvastatin", "Statin", 40.0, 24.0));

        profile.add_condition(condition("Type 2 Diabetes Mellitus", "E11.65", ConditionSeverity::Moderate));
        profile.add_condition(condition("Essential Hypertension", "I10", ConditionSeverity::Moderate));

        let guidelines = vec![
            make_guideline("ADA T2DM 2024", vec![
                med("metformin", "Antidiabetic", 1000.0, 12.0),
                med("glipizide", "Sulfonylurea", 5.0, 24.0),
            ]),
            make_guideline("JNC-8 HTN", vec![
                med("lisinopril", "ACE Inhibitor", 20.0, 24.0),
                med("amlodipine", "CCB", 10.0, 24.0),
            ]),
        ];

        BenchmarkSetup::new(profile)
            .with_guidelines(guidelines)
            .with_time_horizon(168.0)
    }

    /// AF + HF + HTN — cardiac-focused polypharmacy.
    pub fn cardiac_polypharmacy() -> BenchmarkSetup {
        let mut profile = PatientProfile::new(PatientId::new(), PatientInfo {
            age_years: 70.0,
            weight_kg: 78.0,
            height_cm: 172.0,
            sex: Sex::Male,
            serum_creatinine: 1.3,
            ..PatientInfo::default()
        });

        profile.add_medication(med("warfarin", "Anticoagulant", 5.0, 24.0));
        profile.add_medication(med("amiodarone", "Antiarrhythmic", 200.0, 24.0));
        profile.add_medication(med("digoxin", "Cardiac Glycoside", 0.125, 24.0));
        profile.add_medication(med("metoprolol", "Beta Blocker", 50.0, 12.0));
        profile.add_medication(med("lisinopril", "ACE Inhibitor", 10.0, 24.0));
        profile.add_medication(med("furosemide", "Loop Diuretic", 40.0, 12.0));
        profile.add_medication(med("spironolactone", "K-sparing Diuretic", 25.0, 24.0));
        profile.add_medication(med("atorvastatin", "Statin", 20.0, 24.0));

        profile.add_condition(condition("Atrial Fibrillation", "I48.91", ConditionSeverity::Moderate));
        profile.add_condition(condition("Heart Failure", "I50.9", ConditionSeverity::Severe));
        profile.add_condition(condition("Hypertension", "I10", ConditionSeverity::Moderate));

        let guidelines = vec![
            make_guideline("AHA/ACC AF 2023", vec![
                med("warfarin", "Anticoagulant", 5.0, 24.0),
                med("amiodarone", "Antiarrhythmic", 200.0, 24.0),
            ]),
            make_guideline("AHA/ACC HF 2022", vec![
                med("metoprolol", "Beta Blocker", 50.0, 12.0),
                med("lisinopril", "ACE Inhibitor", 10.0, 24.0),
                med("spironolactone", "K-sparing Diuretic", 25.0, 24.0),
            ]),
        ];

        BenchmarkSetup::new(profile)
            .with_guidelines(guidelines)
            .with_properties(vec![
                "no_therapeutic_window_violation".into(),
                "warfarin_inr_range".into(),
                "no_qt_prolongation_combo".into(),
                "digoxin_level_safe".into(),
            ])
            .with_time_horizon(168.0)
    }

    /// Chronic pain + depression — opioid + SSRI interactions.
    pub fn pain_depression() -> BenchmarkSetup {
        let mut profile = PatientProfile::new(PatientId::new(), PatientInfo {
            age_years: 52.0,
            weight_kg: 68.0,
            height_cm: 165.0,
            sex: Sex::Female,
            serum_creatinine: 0.8,
            ..PatientInfo::default()
        });

        profile.add_medication(med("tramadol", "Opioid", 50.0, 6.0));
        profile.add_medication(med("fluoxetine", "SSRI", 40.0, 24.0));
        profile.add_medication(med("gabapentin", "Anticonvulsant", 300.0, 8.0));
        profile.add_medication(med("ibuprofen", "NSAID", 400.0, 8.0));
        profile.add_medication(med("omeprazole", "PPI", 20.0, 24.0));
        profile.add_medication(med("cyclobenzaprine", "Muscle Relaxant", 10.0, 8.0));

        profile.add_condition(condition("Chronic Pain Syndrome", "G89.29", ConditionSeverity::Severe));
        profile.add_condition(condition("Major Depressive Disorder", "F33.1", ConditionSeverity::Moderate));

        let guidelines = vec![
            make_guideline("APA Depression 2023", vec![
                med("fluoxetine", "SSRI", 40.0, 24.0),
            ]),
            make_guideline("Chronic Pain Management", vec![
                med("tramadol", "Opioid", 50.0, 6.0),
                med("gabapentin", "Anticonvulsant", 300.0, 8.0),
            ]),
        ];

        BenchmarkSetup::new(profile)
            .with_guidelines(guidelines)
            .with_properties(vec![
                "no_serotonin_syndrome".into(),
                "no_cns_depression_combo".into(),
                "opioid_safe_dosing".into(),
            ])
            .with_time_horizon(168.0)
    }

    /// Transplant patient — immunosuppressants + multiple interactions.
    pub fn transplant_patient() -> BenchmarkSetup {
        let mut profile = PatientProfile::new(PatientId::new(), PatientInfo {
            age_years: 45.0,
            weight_kg: 70.0,
            height_cm: 175.0,
            sex: Sex::Male,
            serum_creatinine: 1.5,
            ..PatientInfo::default()
        });

        profile.add_medication(med("tacrolimus", "Immunosuppressant", 3.0, 12.0));
        profile.add_medication(med("mycophenolate", "Immunosuppressant", 500.0, 12.0));
        profile.add_medication(med("prednisone", "Corticosteroid", 5.0, 24.0));
        profile.add_medication(med("amlodipine", "CCB", 5.0, 24.0));
        profile.add_medication(med("omeprazole", "PPI", 20.0, 24.0));
        profile.add_medication(med("fluconazole", "Antifungal", 100.0, 24.0));
        profile.add_medication(med("valganciclovir", "Antiviral", 450.0, 24.0));
        profile.add_medication(med("trimethoprim", "Antibiotic", 160.0, 24.0));
        profile.add_medication(med("atorvastatin", "Statin", 10.0, 24.0));

        profile.add_condition(condition("Kidney Transplant", "Z94.0", ConditionSeverity::Severe));
        profile.add_condition(condition("Hypertension", "I10", ConditionSeverity::Moderate));
        profile.add_condition(condition("CMV Prophylaxis", "B25.9", ConditionSeverity::Mild));

        let guidelines = vec![
            make_guideline("KDIGO Transplant 2023", vec![
                med("tacrolimus", "Immunosuppressant", 3.0, 12.0),
                med("mycophenolate", "Immunosuppressant", 500.0, 12.0),
                med("prednisone", "Corticosteroid", 5.0, 24.0),
            ]),
        ];

        BenchmarkSetup::new(profile)
            .with_guidelines(guidelines)
            .with_properties(vec![
                "tacrolimus_level_safe".into(),
                "no_nephrotoxicity_combo".into(),
                "cyp3a4_interaction_check".into(),
            ])
            .with_time_horizon(336.0) // 2 weeks
    }

    /// CKD + multiple renally-cleared drugs.
    pub fn renal_impairment() -> BenchmarkSetup {
        let mut profile = PatientProfile::new(PatientId::new(), PatientInfo {
            age_years: 68.0,
            weight_kg: 75.0,
            height_cm: 168.0,
            sex: Sex::Female,
            serum_creatinine: 2.5,
            ..PatientInfo::default()
        });
        profile.renal_function = RenalFunction::Severe;

        profile.add_medication(med("metformin", "Antidiabetic", 500.0, 24.0));
        profile.add_medication(med("lisinopril", "ACE Inhibitor", 5.0, 24.0));
        profile.add_medication(med("furosemide", "Loop Diuretic", 80.0, 12.0));
        profile.add_medication(med("allopurinol", "Xanthine Oxidase Inhibitor", 100.0, 24.0));
        profile.add_medication(med("digoxin", "Cardiac Glycoside", 0.0625, 24.0));
        profile.add_medication(med("gabapentin", "Anticonvulsant", 100.0, 24.0));
        profile.add_medication(med("spironolactone", "K-sparing Diuretic", 12.5, 24.0));

        profile.add_condition(condition("CKD Stage 4", "N18.4", ConditionSeverity::Severe));
        profile.add_condition(condition("Heart Failure", "I50.9", ConditionSeverity::Moderate));
        profile.add_condition(condition("Gout", "M10.9", ConditionSeverity::Mild));
        profile.add_condition(condition("Type 2 Diabetes", "E11.65", ConditionSeverity::Moderate));

        let labs = vec![
            lab("eGFR", 22.0, "mL/min/1.73m2", 60.0, 120.0),
            lab("Potassium", 5.3, "mEq/L", 3.5, 5.0),
            lab("Creatinine", 2.5, "mg/dL", 0.6, 1.2),
            lab("BUN", 45.0, "mg/dL", 7.0, 20.0),
        ];
        for (name, lv) in labs {
            profile.add_lab_value(&name, lv);
        }

        let guidelines = vec![
            make_guideline("KDIGO CKD", vec![
                med("lisinopril", "ACE Inhibitor", 5.0, 24.0),
            ]),
        ];

        BenchmarkSetup::new(profile)
            .with_guidelines(guidelines)
            .with_properties(vec![
                "renal_dose_adjustment".into(),
                "no_hyperkalemia_risk".into(),
                "metformin_egfr_check".into(),
            ])
            .with_time_horizon(168.0)
    }

    /// Return all common scenarios as (name, setup) pairs.
    pub fn all() -> Vec<(&'static str, BenchmarkSetup)> {
        vec![
            ("Elderly Multimorbid", Self::elderly_multimorbid()),
            ("Diabetes + Hypertension", Self::diabetes_hypertension()),
            ("Cardiac Polypharmacy", Self::cardiac_polypharmacy()),
            ("Pain + Depression", Self::pain_depression()),
            ("Transplant Patient", Self::transplant_patient()),
            ("Renal Impairment", Self::renal_impairment()),
        ]
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Random Scenario Generator
// ═══════════════════════════════════════════════════════════════════════════

/// Simple deterministic PRNG for scenario generation.
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed.wrapping_add(1) }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state
    }

    fn next_usize(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound as u64) as usize
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn next_range_f64(&mut self, lo: f64, hi: f64) -> f64 {
        lo + self.next_f64() * (hi - lo)
    }
}

/// Configurable random scenario generator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomScenarioGenerator {
    pub seed: u64,
    pub min_drugs: usize,
    pub max_drugs: usize,
    pub min_conditions: usize,
    pub max_conditions: usize,
    pub min_age: f64,
    pub max_age: f64,
    pub include_renal_impairment: bool,
    pub include_hepatic_impairment: bool,
}

impl Default for RandomScenarioGenerator {
    fn default() -> Self {
        Self {
            seed: 42,
            min_drugs: 3,
            max_drugs: 15,
            min_conditions: 1,
            max_conditions: 5,
            min_age: 18.0,
            max_age: 95.0,
            include_renal_impairment: true,
            include_hepatic_impairment: false,
        }
    }
}

impl RandomScenarioGenerator {
    pub fn new(seed: u64) -> Self {
        Self { seed, ..Self::default() }
    }

    /// Generate a random patient profile.
    pub fn random_patient(&self) -> PatientProfile {
        let mut rng = SimpleRng::new(self.seed);
        Self::random_patient_with_rng(&mut rng, self)
    }

    fn random_patient_with_rng(rng: &mut SimpleRng, cfg: &RandomScenarioGenerator) -> PatientProfile {
        let age = rng.next_range_f64(cfg.min_age, cfg.max_age);
        let sex = if rng.next_u64() % 2 == 0 { Sex::Male } else { Sex::Female };
        let weight = rng.next_range_f64(50.0, 120.0);
        let height = rng.next_range_f64(150.0, 195.0);
        let scr = rng.next_range_f64(0.6, 3.0);

        let info = PatientInfo {
            age_years: age,
            weight_kg: weight,
            height_cm: height,
            sex,
            serum_creatinine: scr,
            albumin: Some(rng.next_range_f64(2.5, 5.0)),
            bilirubin: Some(rng.next_range_f64(0.3, 2.0)),
            inr: Some(rng.next_range_f64(0.9, 4.0)),
            encephalopathy_grade: Some(0),
            ascites: Some(AscitesGrade::None),
        };

        let mut profile = PatientProfile::new(PatientId::new(), info);

        // Random conditions.
        let all_conditions = vec![
            ("Hypertension", "I10", ConditionSeverity::Moderate),
            ("Type 2 Diabetes", "E11.65", ConditionSeverity::Moderate),
            ("Atrial Fibrillation", "I48.91", ConditionSeverity::Moderate),
            ("Heart Failure", "I50.9", ConditionSeverity::Severe),
            ("COPD", "J44.1", ConditionSeverity::Moderate),
            ("Osteoarthritis", "M17.11", ConditionSeverity::Mild),
            ("Depression", "F33.1", ConditionSeverity::Moderate),
            ("CKD Stage 3", "N18.3", ConditionSeverity::Moderate),
            ("Hypothyroidism", "E03.9", ConditionSeverity::Mild),
            ("GERD", "K21.0", ConditionSeverity::Mild),
        ];

        let n_conditions = cfg.min_conditions + rng.next_usize(cfg.max_conditions - cfg.min_conditions + 1);
        let n_conditions = n_conditions.min(all_conditions.len());
        let mut chosen_conditions = all_conditions.clone();
        // Fisher-Yates partial shuffle.
        for i in 0..n_conditions {
            let j = i + rng.next_usize(chosen_conditions.len() - i);
            chosen_conditions.swap(i, j);
        }
        for &(name, icd10, sev) in chosen_conditions.iter().take(n_conditions) {
            profile.add_condition(condition(name, icd10, sev));
        }

        // Random medications.
        let drug_pool = ScenarioGenerator::drug_pool();
        let n_drugs = cfg.min_drugs + rng.next_usize(cfg.max_drugs - cfg.min_drugs + 1);
        let n_drugs = n_drugs.min(drug_pool.len());

        let mut drug_indices: Vec<usize> = (0..drug_pool.len()).collect();
        for i in 0..n_drugs {
            let j = i + rng.next_usize(drug_indices.len() - i);
            drug_indices.swap(i, j);
        }

        for &idx in drug_indices.iter().take(n_drugs) {
            let (name, class, dose, interval) = drug_pool[idx];
            let dose_variation = rng.next_range_f64(0.5, 1.5);
            profile.add_medication(med(name, class, dose * dose_variation, interval));
        }

        if cfg.include_renal_impairment && scr > 1.5 {
            let egfr_approx = (140.0 - age) * weight / (72.0 * scr);
            profile.renal_function = RenalFunction::from_egfr(egfr_approx);
        }

        profile
    }

    /// Generate a random medication list of specific size.
    pub fn random_medication_list(&self, n: usize) -> Vec<ActiveMedication> {
        let mut rng = SimpleRng::new(self.seed);
        let pool = ScenarioGenerator::drug_pool();
        let actual_n = n.min(pool.len());

        let mut indices: Vec<usize> = (0..pool.len()).collect();
        for i in 0..actual_n {
            let j = i + rng.next_usize(indices.len() - i);
            indices.swap(i, j);
        }

        indices
            .iter()
            .take(actual_n)
            .map(|&idx| {
                let (name, class, dose, interval) = pool[idx];
                med(name, class, dose, interval)
            })
            .collect()
    }

    /// Generate a complete BenchmarkSetup with random parameters.
    pub fn random_setup(&self) -> BenchmarkSetup {
        let profile = self.random_patient();
        let n_gl = (profile.medication_count() / 3).max(1);
        let guidelines: Vec<GuidelineReference> = (0..n_gl)
            .map(|i| GuidelineReference::new(&format!("GL-rand-{}", i)))
            .collect();

        BenchmarkSetup::new(profile)
            .with_guidelines(guidelines)
            .with_time_horizon(168.0)
    }

    /// Generate `n` random setups with different seeds.
    pub fn generate_batch(&self, n: usize) -> Vec<BenchmarkSetup> {
        (0..n)
            .map(|i| {
                let gen = RandomScenarioGenerator { seed: self.seed.wrapping_add(i as u64), ..*self };
                gen.random_setup()
            })
            .collect()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Adversarial Scenario
// ═══════════════════════════════════════════════════════════════════════════

/// Generate an adversarial scenario designed to stress-test the verification engine.
///
/// Uses 20 drugs carefully chosen to maximise the number of pairwise interactions
/// and include cascading CYP-mediated effects, QT prolongation combos, and
/// renal/hepatic interactions simultaneously.
pub fn generate_adversarial_scenario() -> BenchmarkSetup {
    let mut profile = PatientProfile::new(PatientId::new(), PatientInfo {
        age_years: 85.0,
        weight_kg: 55.0,
        height_cm: 160.0,
        sex: Sex::Female,
        serum_creatinine: 2.8,
        albumin: Some(2.8),
        bilirubin: Some(2.5),
        inr: Some(3.5),
        encephalopathy_grade: Some(1),
        ascites: Some(AscitesGrade::Mild),
    });

    // 20 drugs with maximal interaction potential.
    let adversarial_drugs = vec![
        med("warfarin", "Anticoagulant", 7.5, 24.0),
        med("amiodarone", "Antiarrhythmic", 200.0, 24.0),
        med("fluconazole", "Antifungal", 200.0, 24.0),
        med("simvastatin", "Statin", 40.0, 24.0),
        med("digoxin", "Cardiac Glycoside", 0.25, 24.0),
        med("verapamil", "CCB", 120.0, 8.0),
        med("metoprolol", "Beta Blocker", 100.0, 12.0),
        med("fluoxetine", "SSRI", 40.0, 24.0),
        med("tramadol", "Opioid", 100.0, 6.0),
        med("ciprofloxacin", "Fluoroquinolone", 500.0, 12.0),
        med("erythromycin", "Macrolide", 500.0, 6.0),
        med("cyclosporine", "Immunosuppressant", 150.0, 12.0),
        med("spironolactone", "K-sparing Diuretic", 50.0, 24.0),
        med("lisinopril", "ACE Inhibitor", 20.0, 24.0),
        med("aspirin", "NSAID", 325.0, 24.0),
        med("metformin", "Antidiabetic", 1000.0, 12.0),
        med("haloperidol", "Antipsychotic", 10.0, 12.0),
        med("clopidogrel", "Antiplatelet", 75.0, 24.0),
        med("diltiazem", "CCB", 180.0, 8.0),
        med("clarithromycin", "Macrolide", 500.0, 12.0),
    ];

    for m in adversarial_drugs {
        profile.add_medication(m);
    }

    // Many comorbidities.
    profile.add_condition(condition("CKD Stage 4", "N18.4", ConditionSeverity::Severe));
    profile.add_condition(condition("Cirrhosis Child-Pugh B", "K74.60", ConditionSeverity::Severe));
    profile.add_condition(condition("Atrial Fibrillation", "I48.91", ConditionSeverity::Moderate));
    profile.add_condition(condition("Heart Failure", "I50.9", ConditionSeverity::Severe));
    profile.add_condition(condition("Type 2 Diabetes", "E11.65", ConditionSeverity::Moderate));
    profile.add_condition(condition("Depression", "F33.1", ConditionSeverity::Moderate));
    profile.add_condition(condition("Chronic Pain", "G89.29", ConditionSeverity::Severe));
    profile.add_condition(condition("Invasive Fungal Infection", "B37.7", ConditionSeverity::Severe));

    profile.renal_function = RenalFunction::Severe;
    profile.hepatic_function = Some(ChildPughClass::B);

    let guidelines = vec![
        make_guideline("Adversarial GL-1", vec![]),
        make_guideline("Adversarial GL-2", vec![]),
        make_guideline("Adversarial GL-3", vec![]),
    ];

    BenchmarkSetup::new(profile)
        .with_guidelines(guidelines)
        .with_properties(vec![
            "no_therapeutic_window_violation".into(),
            "no_critical_interaction".into(),
            "no_qt_prolongation_combo".into(),
            "no_serotonin_syndrome".into(),
            "renal_dose_adjustment".into(),
            "hepatic_dose_adjustment".into(),
            "no_nephrotoxicity_combo".into(),
            "no_bleeding_risk_combo".into(),
        ])
        .with_time_horizon(336.0)
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::baseline::TmrBaseline;

    #[test]
    fn test_elderly_multimorbid() {
        let setup = CommonScenarios::elderly_multimorbid();
        assert_eq!(setup.medication_count(), 12);
        assert!(setup.guideline_count() >= 3);
        assert!(setup.patient_profile.is_elderly());
    }

    #[test]
    fn test_diabetes_hypertension() {
        let setup = CommonScenarios::diabetes_hypertension();
        assert_eq!(setup.medication_count(), 5);
        assert!(setup.patient_profile.has_condition("Type 2 Diabetes Mellitus"));
    }

    #[test]
    fn test_cardiac_polypharmacy() {
        let setup = CommonScenarios::cardiac_polypharmacy();
        assert_eq!(setup.medication_count(), 8);
        let baseline = TmrBaseline::new();
        let result = baseline.check_interactions(&setup.patient_profile.active_medications);
        assert!(result.interaction_count() >= 3, "Cardiac scenario should have many interactions");
    }

    #[test]
    fn test_pain_depression() {
        let setup = CommonScenarios::pain_depression();
        assert!(setup.medication_count() >= 5);
        assert!(setup.properties.iter().any(|p| p.contains("serotonin")));
    }

    #[test]
    fn test_transplant_patient() {
        let setup = CommonScenarios::transplant_patient();
        assert!(setup.medication_count() >= 8);
        let has_tacrolimus = setup.patient_profile.active_medications
            .iter().any(|m| m.drug_id.as_str() == "tacrolimus");
        assert!(has_tacrolimus);
    }

    #[test]
    fn test_renal_impairment() {
        let setup = CommonScenarios::renal_impairment();
        assert_eq!(setup.patient_profile.renal_function, RenalFunction::Severe);
    }

    #[test]
    fn test_all_common_scenarios() {
        let all = CommonScenarios::all();
        assert_eq!(all.len(), 6);
        for (name, setup) in &all {
            assert!(setup.medication_count() >= 3, "Scenario '{}' has too few drugs", name);
        }
    }

    #[test]
    fn test_generate_polypharmacy() {
        let setup = ScenarioGenerator::generate_polypharmacy_scenario(10);
        assert_eq!(setup.medication_count(), 10);
        assert!(setup.guideline_count() >= 1);
    }

    #[test]
    fn test_scenario_severity_levels() {
        let easy = ScenarioGenerator::generate_by_severity(ScenarioSeverity::Easy);
        let hard = ScenarioGenerator::generate_by_severity(ScenarioSeverity::Hard);
        assert!(easy.medication_count() < hard.medication_count());
    }

    #[test]
    fn test_adversarial_scenario() {
        let setup = generate_adversarial_scenario();
        assert_eq!(setup.medication_count(), 20);
        assert_eq!(setup.pair_count(), 190);
        assert!(setup.properties.len() >= 5);
        let baseline = TmrBaseline::new();
        let result = baseline.check_interactions(&setup.patient_profile.active_medications);
        assert!(result.interaction_count() >= 10, "Adversarial should have many interactions, got {}", result.interaction_count());
    }

    #[test]
    fn test_random_patient() {
        let gen = RandomScenarioGenerator::new(123);
        let profile = gen.random_patient();
        assert!(profile.medication_count() >= gen.min_drugs);
        assert!(profile.medication_count() <= gen.max_drugs);
    }

    #[test]
    fn test_random_medication_list() {
        let gen = RandomScenarioGenerator::new(456);
        let meds = gen.random_medication_list(8);
        assert_eq!(meds.len(), 8);
    }

    #[test]
    fn test_random_setup() {
        let gen = RandomScenarioGenerator::new(789);
        let setup = gen.random_setup();
        assert!(setup.medication_count() >= 3);
        assert!(setup.guideline_count() >= 1);
    }

    #[test]
    fn test_random_batch() {
        let gen = RandomScenarioGenerator::new(42);
        let batch = gen.generate_batch(5);
        assert_eq!(batch.len(), 5);
        // Different seeds should give different patient profiles.
        let ages: Vec<f64> = batch.iter().map(|s| s.patient_profile.demographics.age_years).collect();
        let all_same = ages.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10);
        assert!(!all_same, "Different seeds should produce different patients");
    }

    #[test]
    fn test_scenario_severity_display() {
        assert_eq!(format!("{}", ScenarioSeverity::Easy), "Easy");
        assert_eq!(format!("{}", ScenarioSeverity::Adversarial), "Adversarial");
    }
}
