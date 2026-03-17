//! Clinical conditions, comorbidity, and interaction modeling.
//!
//! Defines the [`Condition`] type representing a diagnosed clinical condition,
//! a library of [`CommonConditions`], comorbidity scoring via
//! [`ComorbidityProfile`], and condition–condition interaction detection.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

// ═══════════════════════════════════════════════════════════════════════════
// ConditionSeverity
// ═══════════════════════════════════════════════════════════════════════════

/// Clinical condition severity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ConditionSeverity {
    Mild,
    Moderate,
    Severe,
    Critical,
}

impl Default for ConditionSeverity {
    fn default() -> Self {
        ConditionSeverity::Moderate
    }
}

impl ConditionSeverity {
    /// Numeric weight for scoring algorithms (1–4).
    pub fn weight(&self) -> u32 {
        match self {
            ConditionSeverity::Mild => 1,
            ConditionSeverity::Moderate => 2,
            ConditionSeverity::Severe => 3,
            ConditionSeverity::Critical => 4,
        }
    }

    /// Descriptive label.
    pub fn label(&self) -> &'static str {
        match self {
            ConditionSeverity::Mild => "Mild",
            ConditionSeverity::Moderate => "Moderate",
            ConditionSeverity::Severe => "Severe",
            ConditionSeverity::Critical => "Critical",
        }
    }
}

impl std::fmt::Display for ConditionSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ConditionStatus
// ═══════════════════════════════════════════════════════════════════════════

/// Active/inactive status for conditions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConditionStatus {
    Active,
    Resolved,
    Chronic,
    Remission,
    Recurrent,
}

impl Default for ConditionStatus {
    fn default() -> Self {
        ConditionStatus::Active
    }
}

impl ConditionStatus {
    /// Whether the condition is currently clinically relevant.
    pub fn is_current(&self) -> bool {
        matches!(
            self,
            ConditionStatus::Active | ConditionStatus::Chronic | ConditionStatus::Recurrent
        )
    }

    /// Whether the condition has been resolved or is in remission.
    pub fn is_inactive(&self) -> bool {
        matches!(
            self,
            ConditionStatus::Resolved | ConditionStatus::Remission
        )
    }
}

impl std::fmt::Display for ConditionStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            ConditionStatus::Active => "Active",
            ConditionStatus::Resolved => "Resolved",
            ConditionStatus::Chronic => "Chronic",
            ConditionStatus::Remission => "Remission",
            ConditionStatus::Recurrent => "Recurrent",
        };
        write!(f, "{s}")
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ConditionCategory
// ═══════════════════════════════════════════════════════════════════════════

/// Broad condition categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConditionCategory {
    Cardiovascular,
    Metabolic,
    Respiratory,
    Neurological,
    Infectious,
    Renal,
    Hepatic,
    Musculoskeletal,
    Psychiatric,
    Oncological,
    Endocrine,
    Hematologic,
    Gastrointestinal,
    Dermatologic,
    Other,
}

impl Default for ConditionCategory {
    fn default() -> Self {
        ConditionCategory::Other
    }
}

impl ConditionCategory {
    /// Infer category from an ICD-10 code prefix.
    pub fn from_icd10(code: &str) -> Self {
        let upper = code.to_uppercase();
        if upper.starts_with('I') {
            ConditionCategory::Cardiovascular
        } else if upper.starts_with("E1") || upper.starts_with("E7") || upper.starts_with("E6") {
            ConditionCategory::Metabolic
        } else if upper.starts_with('J') {
            ConditionCategory::Respiratory
        } else if upper.starts_with('G') {
            ConditionCategory::Neurological
        } else if upper.starts_with('A') || upper.starts_with('B') {
            ConditionCategory::Infectious
        } else if upper.starts_with('N') {
            ConditionCategory::Renal
        } else if upper.starts_with("K7") {
            ConditionCategory::Hepatic
        } else if upper.starts_with('M') {
            ConditionCategory::Musculoskeletal
        } else if upper.starts_with('F') {
            ConditionCategory::Psychiatric
        } else if upper.starts_with('C') || upper.starts_with('D') {
            ConditionCategory::Oncological
        } else if upper.starts_with("E0") {
            ConditionCategory::Endocrine
        } else if upper.starts_with('K') {
            ConditionCategory::Gastrointestinal
        } else if upper.starts_with('L') {
            ConditionCategory::Dermatologic
        } else {
            ConditionCategory::Other
        }
    }

    /// Whether this category is organ-system-based (affects PK).
    pub fn affects_pharmacokinetics(&self) -> bool {
        matches!(
            self,
            ConditionCategory::Renal
                | ConditionCategory::Hepatic
                | ConditionCategory::Cardiovascular
                | ConditionCategory::Gastrointestinal
        )
    }
}

impl std::fmt::Display for ConditionCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Condition
// ═══════════════════════════════════════════════════════════════════════════

/// A clinical condition or diagnosis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Condition {
    pub id: String,
    pub code: String,
    pub name: String,
    pub category: ConditionCategory,
    pub severity: ConditionSeverity,
    pub status: ConditionStatus,
    pub onset_date: Option<String>,
    pub resolved_date: Option<String>,
    pub notes: String,
}

impl Condition {
    pub fn new(code: &str, name: &str) -> Self {
        Condition {
            id: format!("cond-{}", code.to_lowercase().replace('.', "")),
            code: code.to_string(),
            name: name.to_string(),
            category: ConditionCategory::from_icd10(code),
            severity: ConditionSeverity::default(),
            status: ConditionStatus::Active,
            onset_date: None,
            resolved_date: None,
            notes: String::new(),
        }
    }

    pub fn with_category(mut self, cat: ConditionCategory) -> Self {
        self.category = cat;
        self
    }

    pub fn with_severity(mut self, sev: ConditionSeverity) -> Self {
        self.severity = sev;
        self
    }

    pub fn with_status(mut self, status: ConditionStatus) -> Self {
        self.status = status;
        self
    }

    pub fn with_onset(mut self, date: &str) -> Self {
        self.onset_date = Some(date.to_string());
        self
    }

    pub fn with_notes(mut self, notes: &str) -> Self {
        self.notes = notes.to_string();
        self
    }

    /// Whether this condition is currently active.
    pub fn is_active(&self) -> bool {
        self.status.is_current()
    }

    /// Whether this condition matches an ICD-10 prefix (e.g., "I1" for
    /// hypertensive diseases).
    pub fn matches_icd10_prefix(&self, prefix: &str) -> bool {
        self.code
            .to_uppercase()
            .starts_with(&prefix.to_uppercase())
    }

    /// Charlson Comorbidity Index weight for this condition.
    pub fn charlson_weight(&self) -> u32 {
        let code = self.code.to_uppercase();
        // Weights based on standard Charlson mapping
        if code.starts_with("I21") || code.starts_with("I22") {
            // MI
            1
        } else if code.starts_with("I50") {
            // CHF
            1
        } else if code.starts_with("I7") {
            // PVD
            1
        } else if code.starts_with("I6") || code.starts_with("G45") {
            // CVA/TIA
            1
        } else if code.starts_with("F0") || code.starts_with("G30") {
            // Dementia
            1
        } else if code.starts_with("J4") {
            // COPD/Asthma
            1
        } else if code.starts_with("M0") || code.starts_with("M3") {
            // Connective tissue
            1
        } else if code.starts_with("K25") || code.starts_with("K26") || code.starts_with("K27") {
            // Peptic ulcer
            1
        } else if code.starts_with("K7") {
            // Liver disease
            if code.starts_with("K74") || code.starts_with("K72") {
                3 // Severe
            } else {
                1 // Mild
            }
        } else if code.starts_with("E1") {
            // Diabetes
            if self.notes.to_lowercase().contains("end-organ") {
                2 // With end-organ damage
            } else {
                1
            }
        } else if code.starts_with("G8") || code.starts_with("G82") {
            // Hemiplegia/paraplegia
            2
        } else if code.starts_with("N18") {
            // CKD
            2
        } else if code.starts_with("C") {
            // Malignancy
            if self.notes.to_lowercase().contains("metastatic") {
                6
            } else {
                2
            }
        } else if code.starts_with("B20") {
            // HIV/AIDS
            6
        } else {
            0
        }
    }
}

impl std::fmt::Display for Condition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} [{}] ({})", self.name, self.code, self.status)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CommonConditions
// ═══════════════════════════════════════════════════════════════════════════

/// Well-known conditions for quick construction.
pub struct CommonConditions;

impl CommonConditions {
    pub fn hypertension() -> Condition {
        Condition::new("I10", "Essential Hypertension")
            .with_category(ConditionCategory::Cardiovascular)
    }

    pub fn diabetes_type2() -> Condition {
        Condition::new("E11", "Type 2 Diabetes Mellitus")
            .with_category(ConditionCategory::Metabolic)
    }

    pub fn atrial_fibrillation() -> Condition {
        Condition::new("I48", "Atrial Fibrillation")
            .with_category(ConditionCategory::Cardiovascular)
    }

    pub fn heart_failure() -> Condition {
        Condition::new("I50", "Heart Failure")
            .with_category(ConditionCategory::Cardiovascular)
    }

    pub fn depression() -> Condition {
        Condition::new("F32", "Major Depressive Disorder")
            .with_category(ConditionCategory::Psychiatric)
    }

    pub fn copd() -> Condition {
        Condition::new("J44", "COPD")
            .with_category(ConditionCategory::Respiratory)
    }

    pub fn ckd() -> Condition {
        Condition::new("N18", "Chronic Kidney Disease")
            .with_category(ConditionCategory::Renal)
    }

    pub fn asthma() -> Condition {
        Condition::new("J45", "Asthma")
            .with_category(ConditionCategory::Respiratory)
    }

    pub fn dyslipidemia() -> Condition {
        Condition::new("E78", "Dyslipidemia")
            .with_category(ConditionCategory::Metabolic)
    }

    pub fn hypothyroidism() -> Condition {
        Condition::new("E03", "Hypothyroidism")
            .with_category(ConditionCategory::Endocrine)
    }

    pub fn gerd() -> Condition {
        Condition::new("K21", "Gastroesophageal Reflux Disease")
            .with_category(ConditionCategory::Gastrointestinal)
    }

    pub fn osteoporosis() -> Condition {
        Condition::new("M81", "Osteoporosis")
            .with_category(ConditionCategory::Musculoskeletal)
    }

    pub fn obesity() -> Condition {
        Condition::new("E66", "Obesity")
            .with_category(ConditionCategory::Metabolic)
    }

    pub fn anxiety() -> Condition {
        Condition::new("F41", "Generalized Anxiety Disorder")
            .with_category(ConditionCategory::Psychiatric)
    }

    pub fn chronic_pain() -> Condition {
        Condition::new("G89", "Chronic Pain Syndrome")
            .with_category(ConditionCategory::Neurological)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ConditionInteraction
// ═══════════════════════════════════════════════════════════════════════════

/// The clinical significance of a condition–condition interaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InteractionSignificance {
    /// Synergistic worsening (e.g., HF + CKD).
    Synergistic,
    /// One condition complicates management of the other.
    Complicating,
    /// Shared risk pathway elevates combined risk.
    SharedRisk,
    /// Primarily informational.
    Informational,
}

/// Interaction between two conditions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConditionInteraction {
    pub condition_a: String,
    pub condition_b: String,
    pub significance: InteractionSignificance,
    pub effect: String,
    pub management_note: String,
}

impl ConditionInteraction {
    pub fn new(a: &str, b: &str, sig: InteractionSignificance, effect: &str) -> Self {
        Self {
            condition_a: a.to_string(),
            condition_b: b.to_string(),
            significance: sig,
            effect: effect.to_string(),
            management_note: String::new(),
        }
    }

    pub fn with_management(mut self, note: &str) -> Self {
        self.management_note = note.to_string();
        self
    }

    /// Whether this interaction involves a given condition code.
    pub fn involves(&self, code: &str) -> bool {
        self.condition_a.eq_ignore_ascii_case(code)
            || self.condition_b.eq_ignore_ascii_case(code)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ConditionInteractionDatabase
// ═══════════════════════════════════════════════════════════════════════════

/// A small built-in database of known condition–condition interactions.
pub struct ConditionInteractionDatabase;

impl ConditionInteractionDatabase {
    /// Return all known interactions between the supplied conditions.
    pub fn find_interactions(conditions: &[Condition]) -> Vec<ConditionInteraction> {
        let mut result = Vec::new();
        let codes: Vec<&str> = conditions
            .iter()
            .filter(|c| c.is_active())
            .map(|c| c.code.as_str())
            .collect();

        let known = Self::known_pairs();
        for (a_prefix, b_prefix, sig, effect, mgmt) in known {
            let has_a = codes.iter().any(|c| c.starts_with(a_prefix));
            let has_b = codes.iter().any(|c| c.starts_with(b_prefix));
            if has_a && has_b {
                result.push(
                    ConditionInteraction::new(a_prefix, b_prefix, sig, effect)
                        .with_management(mgmt),
                );
            }
        }
        result
    }

    fn known_pairs() -> Vec<(&'static str, &'static str, InteractionSignificance, &'static str, &'static str)> {
        vec![
            ("I50", "N18", InteractionSignificance::Synergistic,
             "Heart failure and CKD worsen each other",
             "Careful diuretic dosing; avoid nephrotoxins"),
            ("I10", "N18", InteractionSignificance::Complicating,
             "Hypertension accelerates CKD progression",
             "Tight BP control (<130/80); prefer ACE/ARB"),
            ("E11", "N18", InteractionSignificance::Synergistic,
             "Diabetic nephropathy pathway",
             "SGLT2 inhibitors have renal benefit"),
            ("E11", "I10", InteractionSignificance::SharedRisk,
             "Shared cardiovascular risk",
             "Target BP <130/80; consider SGLT2i"),
            ("I48", "I50", InteractionSignificance::Synergistic,
             "AF worsens heart failure hemodynamics",
             "Rate control essential; anticoagulation"),
            ("F32", "G89", InteractionSignificance::Complicating,
             "Depression amplifies chronic pain perception",
             "Consider SNRIs for dual benefit"),
            ("J44", "I50", InteractionSignificance::Complicating,
             "COPD exacerbations worsen HF",
             "Avoid non-selective beta-blockers"),
            ("E11", "E66", InteractionSignificance::SharedRisk,
             "Obesity worsens insulin resistance",
             "Weight management; consider GLP-1 RA"),
            ("E11", "E78", InteractionSignificance::SharedRisk,
             "Metabolic syndrome cluster",
             "Statin therapy indicated"),
            ("I10", "I50", InteractionSignificance::Synergistic,
             "Uncontrolled HTN is leading cause of HF",
             "Aggressive BP management; ACEi + BB + diuretic"),
        ]
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ComorbidityProfile
// ═══════════════════════════════════════════════════════════════════════════

/// Comorbidity profile summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComorbidityProfile {
    pub conditions: Vec<Condition>,
    pub charlson_index: u32,
    pub category_counts: HashMap<String, usize>,
    pub interactions: Vec<ConditionInteraction>,
    pub pk_relevant_count: usize,
}

impl Default for ComorbidityProfile {
    fn default() -> Self {
        ComorbidityProfile {
            conditions: Vec::new(),
            charlson_index: 0,
            category_counts: HashMap::new(),
            interactions: Vec::new(),
            pk_relevant_count: 0,
        }
    }
}

impl ComorbidityProfile {
    /// Build a complete comorbidity profile from a set of conditions.
    pub fn from_conditions(conditions: Vec<Condition>) -> Self {
        let charlson_index: u32 = conditions
            .iter()
            .filter(|c| c.is_active())
            .map(|c| c.charlson_weight())
            .sum();

        let mut category_counts: HashMap<String, usize> = HashMap::new();
        for c in conditions.iter().filter(|c| c.is_active()) {
            *category_counts
                .entry(c.category.to_string())
                .or_insert(0) += 1;
        }

        let pk_relevant_count = conditions
            .iter()
            .filter(|c| c.is_active() && c.category.affects_pharmacokinetics())
            .count();

        let interactions = ConditionInteractionDatabase::find_interactions(&conditions);

        ComorbidityProfile {
            conditions,
            charlson_index,
            category_counts,
            interactions,
            pk_relevant_count,
        }
    }

    /// Number of active conditions.
    pub fn active_count(&self) -> usize {
        self.conditions.iter().filter(|c| c.is_active()).count()
    }

    /// Unique categories of active conditions.
    pub fn active_categories(&self) -> HashSet<ConditionCategory> {
        self.conditions
            .iter()
            .filter(|c| c.is_active())
            .map(|c| c.category)
            .collect()
    }

    /// Whether the profile has multi-system involvement (≥ 3 categories).
    pub fn is_multisystem(&self) -> bool {
        self.active_categories().len() >= 3
    }

    /// Estimated 10-year mortality risk based on Charlson index and age.
    pub fn estimated_10yr_mortality(&self, age: f64) -> f64 {
        let age_score = if age >= 80.0 {
            4
        } else if age >= 70.0 {
            3
        } else if age >= 60.0 {
            2
        } else if age >= 50.0 {
            1
        } else {
            0
        };
        let combined = self.charlson_index + age_score;
        // Charlson approximation: 10-year mortality %
        match combined {
            0 => 0.12,
            1 => 0.26,
            2 => 0.52,
            3 => 0.85,
            4..=5 => 0.93,
            _ => 0.98,
        }
    }

    /// Summary string for display.
    pub fn summary(&self) -> String {
        format!(
            "{} active conditions, Charlson index {}, {} interactions, {} PK-relevant",
            self.active_count(),
            self.charlson_index,
            self.interactions.len(),
            self.pk_relevant_count
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_condition_new() {
        let c = Condition::new("I10", "Hypertension");
        assert_eq!(c.code, "I10");
        assert_eq!(c.name, "Hypertension");
        assert_eq!(c.category, ConditionCategory::Cardiovascular);
        assert!(c.is_active());
    }

    #[test]
    fn test_condition_builder_chain() {
        let c = Condition::new("E11", "DM2")
            .with_severity(ConditionSeverity::Severe)
            .with_status(ConditionStatus::Chronic)
            .with_onset("2020-01-15")
            .with_notes("Insulin-dependent");
        assert_eq!(c.severity, ConditionSeverity::Severe);
        assert_eq!(c.status, ConditionStatus::Chronic);
        assert!(c.is_active());
        assert_eq!(c.onset_date.as_deref(), Some("2020-01-15"));
    }

    #[test]
    fn test_condition_status_current() {
        assert!(ConditionStatus::Active.is_current());
        assert!(ConditionStatus::Chronic.is_current());
        assert!(ConditionStatus::Recurrent.is_current());
        assert!(!ConditionStatus::Resolved.is_current());
        assert!(!ConditionStatus::Remission.is_current());
    }

    #[test]
    fn test_condition_icd10_prefix() {
        let c = Condition::new("I48.1", "Persistent AF");
        assert!(c.matches_icd10_prefix("I48"));
        assert!(c.matches_icd10_prefix("I4"));
        assert!(!c.matches_icd10_prefix("E1"));
    }

    #[test]
    fn test_category_from_icd10() {
        assert_eq!(
            ConditionCategory::from_icd10("I10"),
            ConditionCategory::Cardiovascular
        );
        assert_eq!(
            ConditionCategory::from_icd10("E11"),
            ConditionCategory::Metabolic
        );
        assert_eq!(
            ConditionCategory::from_icd10("J44"),
            ConditionCategory::Respiratory
        );
        assert_eq!(
            ConditionCategory::from_icd10("N18"),
            ConditionCategory::Renal
        );
        assert_eq!(
            ConditionCategory::from_icd10("F32"),
            ConditionCategory::Psychiatric
        );
    }

    #[test]
    fn test_category_affects_pk() {
        assert!(ConditionCategory::Renal.affects_pharmacokinetics());
        assert!(ConditionCategory::Hepatic.affects_pharmacokinetics());
        assert!(!ConditionCategory::Psychiatric.affects_pharmacokinetics());
    }

    #[test]
    fn test_severity_weight() {
        assert_eq!(ConditionSeverity::Mild.weight(), 1);
        assert_eq!(ConditionSeverity::Critical.weight(), 4);
        assert!(ConditionSeverity::Severe > ConditionSeverity::Moderate);
    }

    #[test]
    fn test_common_conditions() {
        let htn = CommonConditions::hypertension();
        assert_eq!(htn.code, "I10");
        assert_eq!(htn.category, ConditionCategory::Cardiovascular);

        let dm = CommonConditions::diabetes_type2();
        assert_eq!(dm.code, "E11");

        let ckd = CommonConditions::ckd();
        assert_eq!(ckd.code, "N18");
        assert_eq!(ckd.category, ConditionCategory::Renal);
    }

    #[test]
    fn test_charlson_weight() {
        let chf = Condition::new("I50", "CHF");
        assert_eq!(chf.charlson_weight(), 1);

        let ckd = Condition::new("N18", "CKD");
        assert_eq!(ckd.charlson_weight(), 2);

        let cancer = Condition::new("C34", "Lung Cancer")
            .with_notes("metastatic disease");
        assert_eq!(cancer.charlson_weight(), 6);
    }

    #[test]
    fn test_comorbidity_profile() {
        let conditions = vec![
            CommonConditions::hypertension(),
            CommonConditions::diabetes_type2(),
            CommonConditions::heart_failure(),
            CommonConditions::ckd(),
        ];
        let profile = ComorbidityProfile::from_conditions(conditions);
        assert_eq!(profile.active_count(), 4);
        assert!(profile.charlson_index >= 4);
        assert!(!profile.interactions.is_empty());
        assert!(profile.pk_relevant_count >= 2);
    }

    #[test]
    fn test_comorbidity_multisystem() {
        let conditions = vec![
            CommonConditions::hypertension(),
            CommonConditions::diabetes_type2(),
            CommonConditions::depression(),
        ];
        let profile = ComorbidityProfile::from_conditions(conditions);
        assert!(profile.is_multisystem());
    }

    #[test]
    fn test_comorbidity_single_system() {
        let conditions = vec![
            CommonConditions::hypertension(),
            CommonConditions::atrial_fibrillation(),
        ];
        let profile = ComorbidityProfile::from_conditions(conditions);
        assert!(!profile.is_multisystem());
    }

    #[test]
    fn test_interaction_detection() {
        let conditions = vec![
            CommonConditions::heart_failure(),
            CommonConditions::ckd(),
        ];
        let interactions = ConditionInteractionDatabase::find_interactions(&conditions);
        assert!(!interactions.is_empty());
        assert!(interactions[0].involves("I50"));
    }

    #[test]
    fn test_no_interactions() {
        let conditions = vec![
            CommonConditions::osteoporosis(),
            CommonConditions::hypothyroidism(),
        ];
        let interactions = ConditionInteractionDatabase::find_interactions(&conditions);
        assert!(interactions.is_empty());
    }

    #[test]
    fn test_estimated_mortality() {
        let conditions = vec![
            CommonConditions::hypertension(),
            CommonConditions::diabetes_type2(),
            CommonConditions::ckd(),
        ];
        let profile = ComorbidityProfile::from_conditions(conditions);
        let mort = profile.estimated_10yr_mortality(75.0);
        assert!(mort > 0.0 && mort <= 1.0);
    }

    #[test]
    fn test_condition_display() {
        let c = Condition::new("I10", "Hypertension");
        let s = format!("{c}");
        assert!(s.contains("Hypertension"));
        assert!(s.contains("I10"));
        assert!(s.contains("Active"));
    }

    #[test]
    fn test_resolved_condition_excluded_from_active() {
        let conditions = vec![
            CommonConditions::hypertension(),
            Condition::new("A09", "Gastroenteritis").with_status(ConditionStatus::Resolved),
        ];
        let profile = ComorbidityProfile::from_conditions(conditions);
        assert_eq!(profile.active_count(), 1);
    }
}
