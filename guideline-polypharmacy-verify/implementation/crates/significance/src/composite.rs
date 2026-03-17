//! Composite significance scoring engine.
//!
//! Combines DrugBank severity, Beers Criteria violations, FAERS disproportionality
//! signals, and comorbidity prevalence into a single weighted score. Classifies
//! each conflict into a [`ClinicalSeverity`] tier and produces ranked output.

use serde::{Deserialize, Serialize};
use std::fmt;

use crate::beers::BeersCriteria;
use crate::comorbidity::{
    compute_comorbidity_component, MedicarePrevalenceData,
};
use crate::drugbank::DrugBankDatabase;
use crate::faers::FaersDatabase;
use crate::{ClinicalConfig, ConfirmedConflict, Medication, PatientProfile};
use crate::Severity;

// ─────────────────────────── ClinicalSeverity ────────────────────────────

/// Final severity classification after composite scoring.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub enum ClinicalSeverity {
    Negligible,
    Minor,
    Moderate,
    Significant,
    Critical,
}

impl ClinicalSeverity {
    /// Numeric floor for each classification tier.
    pub fn threshold(&self) -> f64 {
        match self {
            ClinicalSeverity::Negligible => 0.0,
            ClinicalSeverity::Minor => 0.2,
            ClinicalSeverity::Moderate => 0.4,
            ClinicalSeverity::Significant => 0.6,
            ClinicalSeverity::Critical => 0.8,
        }
    }

    /// Whether this severity is at least the given level.
    pub fn at_least(&self, other: ClinicalSeverity) -> bool {
        *self >= other
    }

    /// All tiers in ascending order.
    pub fn all() -> &'static [ClinicalSeverity] {
        &[
            ClinicalSeverity::Negligible,
            ClinicalSeverity::Minor,
            ClinicalSeverity::Moderate,
            ClinicalSeverity::Significant,
            ClinicalSeverity::Critical,
        ]
    }
}

impl fmt::Display for ClinicalSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ClinicalSeverity::Negligible => write!(f, "Negligible"),
            ClinicalSeverity::Minor => write!(f, "Minor"),
            ClinicalSeverity::Moderate => write!(f, "Moderate"),
            ClinicalSeverity::Significant => write!(f, "Significant"),
            ClinicalSeverity::Critical => write!(f, "Critical"),
        }
    }
}

/// Classify a total score into a ClinicalSeverity tier.
pub fn classify_severity(score: f64) -> ClinicalSeverity {
    if score >= 0.8 {
        ClinicalSeverity::Critical
    } else if score >= 0.6 {
        ClinicalSeverity::Significant
    } else if score >= 0.4 {
        ClinicalSeverity::Moderate
    } else if score >= 0.2 {
        ClinicalSeverity::Minor
    } else {
        ClinicalSeverity::Negligible
    }
}

// ─────────────────────────── SignificanceScore ────────────────────────────

/// Breakdown of a significance score into its components.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignificanceScore {
    pub total_score: f64,
    pub drugbank_component: f64,
    pub beers_component: f64,
    pub faers_component: f64,
    pub comorbidity_component: f64,
    pub severity_classification: ClinicalSeverity,
    /// Raw base severity from the conflict itself.
    pub base_severity_score: f64,
}

impl SignificanceScore {
    pub fn zero() -> Self {
        SignificanceScore {
            total_score: 0.0,
            drugbank_component: 0.0,
            beers_component: 0.0,
            faers_component: 0.0,
            comorbidity_component: 0.0,
            severity_classification: ClinicalSeverity::Negligible,
            base_severity_score: 0.0,
        }
    }

    /// Whether the score meets a minimum severity level.
    pub fn meets_threshold(&self, min: ClinicalSeverity) -> bool {
        self.severity_classification.at_least(min)
    }
}

impl fmt::Display for SignificanceScore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:.3} [{}] (DB:{:.2} BEERS:{:.2} FAERS:{:.2} COMOR:{:.2})",
            self.total_score, self.severity_classification,
            self.drugbank_component, self.beers_component,
            self.faers_component, self.comorbidity_component,
        )
    }
}

// ─────────────────────────── ScoredConflict ──────────────────────────────

/// A confirmed conflict paired with its computed significance score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredConflict {
    pub conflict: ConfirmedConflict,
    pub score: SignificanceScore,
    pub rank: usize,
}

impl ScoredConflict {
    pub fn new(conflict: ConfirmedConflict, score: SignificanceScore) -> Self {
        ScoredConflict { conflict, score, rank: 0 }
    }

    pub fn severity(&self) -> ClinicalSeverity {
        self.score.severity_classification
    }

    pub fn total_score(&self) -> f64 {
        self.score.total_score
    }
}

impl fmt::Display for ScoredConflict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "#{} {} ↔ {} — {} (score: {:.3})",
            self.rank,
            self.conflict.drug_a_name,
            self.conflict.drug_b_name,
            self.score.severity_classification,
            self.score.total_score,
        )
    }
}

// ─────────────────────────── SignificanceScorer ──────────────────────────

/// Composite significance scorer that combines all evidence sources.
#[derive(Debug, Clone)]
pub struct SignificanceScorer {
    pub drugbank_weight: f64,
    pub beers_weight: f64,
    pub faers_weight: f64,
    pub comorbidity_weight: f64,
    drugbank: DrugBankDatabase,
    beers: BeersCriteria,
    faers: FaersDatabase,
    prevalence: MedicarePrevalenceData,
}

impl SignificanceScorer {
    /// Create a scorer with custom weights.
    pub fn new(
        drugbank_weight: f64,
        beers_weight: f64,
        faers_weight: f64,
        comorbidity_weight: f64,
    ) -> Self {
        SignificanceScorer {
            drugbank_weight,
            beers_weight,
            faers_weight,
            comorbidity_weight,
            drugbank: DrugBankDatabase::with_defaults(),
            beers: BeersCriteria::with_defaults(),
            faers: FaersDatabase::with_defaults(),
            prevalence: MedicarePrevalenceData::with_defaults(),
        }
    }

    /// Create from a ClinicalConfig.
    pub fn from_config(config: &ClinicalConfig) -> Self {
        SignificanceScorer {
            drugbank_weight: config.drugbank_weight,
            beers_weight: config.beers_weight,
            faers_weight: config.faers_weight,
            comorbidity_weight: config.comorbidity_weight,
            drugbank: DrugBankDatabase::with_defaults(),
            beers: BeersCriteria::with_defaults(),
            faers: FaersDatabase::with_defaults(),
            prevalence: MedicarePrevalenceData::with_defaults(),
        }
    }

    /// Create with default weights (0.35, 0.25, 0.20, 0.20).
    pub fn with_defaults() -> Self {
        Self::from_config(&ClinicalConfig::default())
    }

    /// Replace the DrugBank database.
    pub fn with_drugbank(mut self, db: DrugBankDatabase) -> Self {
        self.drugbank = db;
        self
    }

    /// Replace the FAERS database.
    pub fn with_faers(mut self, db: FaersDatabase) -> Self {
        self.faers = db;
        self
    }

    /// Access the DrugBank database.
    pub fn drugbank(&self) -> &DrugBankDatabase {
        &self.drugbank
    }

    /// Access the FAERS database.
    pub fn faers(&self) -> &FaersDatabase {
        &self.faers
    }

    /// Compute the total weight sum (for normalization).
    fn total_weight(&self) -> f64 {
        let sum = self.drugbank_weight + self.beers_weight + self.faers_weight + self.comorbidity_weight;
        if sum == 0.0 { 1.0 } else { sum }
    }

    // ── Component scoring ───────────────────────────────────────────────

    /// DrugBank component: lookup severity and evidence for the pair.
    fn score_drugbank(&self, conflict: &ConfirmedConflict) -> f64 {
        let db_score = self.drugbank.composite_score(
            conflict.drug_a.as_str(),
            conflict.drug_b.as_str(),
        );
        if db_score > 0.0 {
            return db_score;
        }
        // Fall back to the conflict's own severity rating
        base_severity_score(conflict.severity)
    }

    /// Beers component: check whether the pair triggers Beers violations.
    fn score_beers(&self, conflict: &ConfirmedConflict, patient: &PatientProfile) -> f64 {
        if !patient.is_elderly() {
            return 0.0;
        }

        let med_a = Medication::new(
            &conflict.drug_a_name,
            "",
            0.0,
        );
        let med_b = Medication::new(
            &conflict.drug_b_name,
            "",
            0.0,
        );

        let mut max_score = 0.0_f64;

        // Check individual medications
        let violations_a = self.beers.check_single_medication(&med_a, patient);
        let violations_b = self.beers.check_single_medication(&med_b, patient);

        for v in violations_a.iter().chain(violations_b.iter()) {
            max_score = max_score.max(v.severity_score);
        }

        // Check drug interaction
        let interaction_violations = self.beers.check_drug_interaction(&med_a, &med_b, patient);
        for v in &interaction_violations {
            max_score = max_score.max(v.severity_score);
        }

        max_score
    }

    /// FAERS component: look up disproportionality signals.
    fn score_faers(&self, conflict: &ConfirmedConflict) -> f64 {
        self.faers.max_composite_score(
            conflict.drug_a.as_str(),
            conflict.drug_b.as_str(),
        )
    }

    /// Comorbidity component: prevalence-weighted severity.
    fn score_comorbidity(&self, conflict: &ConfirmedConflict, patient: &PatientProfile) -> f64 {
        let base = base_severity_score(conflict.severity);
        compute_comorbidity_component(patient, base, &self.prevalence)
    }

    // ── Public API ──────────────────────────────────────────────────────

    /// Score a single conflict.
    pub fn score_conflict(
        &self,
        conflict: &ConfirmedConflict,
        patient: &PatientProfile,
    ) -> SignificanceScore {
        let db = self.score_drugbank(conflict);
        let beers = self.score_beers(conflict, patient);
        let faers = self.score_faers(conflict);
        let comorbid = self.score_comorbidity(conflict, patient);

        let total_w = self.total_weight();
        let total = (self.drugbank_weight * db
            + self.beers_weight * beers
            + self.faers_weight * faers
            + self.comorbidity_weight * comorbid)
            / total_w;

        let total_clamped = total.min(1.0).max(0.0);

        SignificanceScore {
            total_score: total_clamped,
            drugbank_component: db,
            beers_component: beers,
            faers_component: faers,
            comorbidity_component: comorbid,
            severity_classification: classify_severity(total_clamped),
            base_severity_score: base_severity_score(conflict.severity),
        }
    }

    /// Score all conflicts and return sorted results.
    pub fn score_all_conflicts(
        &self,
        conflicts: &[ConfirmedConflict],
        patient: &PatientProfile,
    ) -> Vec<ScoredConflict> {
        let mut scored: Vec<ScoredConflict> = conflicts
            .iter()
            .map(|c| {
                let score = self.score_conflict(c, patient);
                ScoredConflict::new(c.clone(), score)
            })
            .collect();

        rank_conflicts(&mut scored);
        scored
    }
}

/// Map the base Severity enum to a 0..1 score.
pub fn base_severity_score(severity: Severity) -> f64 {
    match severity {
        Severity::Minor => 0.2,
        Severity::Moderate => 0.5,
        Severity::Major => 0.8,
        Severity::Contraindicated => 1.0,
    }
}

/// Rank conflicts by descending total score.
pub fn rank_conflicts(scored: &mut [ScoredConflict]) {
    scored.sort_by(|a, b| {
        b.score
            .total_score
            .partial_cmp(&a.score.total_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for (i, s) in scored.iter_mut().enumerate() {
        s.rank = i + 1;
    }
}

/// Filter scored conflicts by a minimum severity classification.
pub fn filter_by_threshold(
    scored: &[ScoredConflict],
    min_severity: ClinicalSeverity,
) -> Vec<ScoredConflict> {
    scored
        .iter()
        .filter(|s| s.severity().at_least(min_severity))
        .cloned()
        .collect()
}

// ─────────────────────────── SignificanceReport ──────────────────────────

/// Summary statistics for a set of scored conflicts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignificanceReport {
    pub total_conflicts: usize,
    pub critical_count: usize,
    pub significant_count: usize,
    pub moderate_count: usize,
    pub minor_count: usize,
    pub negligible_count: usize,
    pub mean_score: f64,
    pub max_score: f64,
    pub min_score: f64,
    pub median_score: f64,
    pub top_conflicts: Vec<ScoredConflict>,
}

impl SignificanceReport {
    /// Generate a report from a list of scored conflicts.
    pub fn generate(conflicts: &[ScoredConflict]) -> Self {
        if conflicts.is_empty() {
            return SignificanceReport {
                total_conflicts: 0,
                critical_count: 0,
                significant_count: 0,
                moderate_count: 0,
                minor_count: 0,
                negligible_count: 0,
                mean_score: 0.0,
                max_score: 0.0,
                min_score: 0.0,
                median_score: 0.0,
                top_conflicts: Vec::new(),
            };
        }

        let mut critical = 0;
        let mut significant = 0;
        let mut moderate = 0;
        let mut minor = 0;
        let mut negligible = 0;
        let mut sum = 0.0_f64;
        let mut max_s = f64::NEG_INFINITY;
        let mut min_s = f64::INFINITY;

        let mut scores: Vec<f64> = Vec::with_capacity(conflicts.len());

        for sc in conflicts {
            let s = sc.total_score();
            scores.push(s);
            sum += s;
            max_s = max_s.max(s);
            min_s = min_s.min(s);

            match sc.severity() {
                ClinicalSeverity::Critical => critical += 1,
                ClinicalSeverity::Significant => significant += 1,
                ClinicalSeverity::Moderate => moderate += 1,
                ClinicalSeverity::Minor => minor += 1,
                ClinicalSeverity::Negligible => negligible += 1,
            }
        }

        scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = if scores.len() % 2 == 0 {
            (scores[scores.len() / 2 - 1] + scores[scores.len() / 2]) / 2.0
        } else {
            scores[scores.len() / 2]
        };

        let n = conflicts.len() as f64;
        let top = conflicts.iter().take(5).cloned().collect();

        SignificanceReport {
            total_conflicts: conflicts.len(),
            critical_count: critical,
            significant_count: significant,
            moderate_count: moderate,
            minor_count: minor,
            negligible_count: negligible,
            mean_score: sum / n,
            max_score: max_s,
            min_score: min_s,
            median_score: median,
            top_conflicts: top,
        }
    }
}

impl fmt::Display for SignificanceReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Significance Report: {} conflicts (Critical:{}, Significant:{}, Moderate:{}, Minor:{}, Negligible:{}) mean={:.3} max={:.3}",
            self.total_conflicts,
            self.critical_count, self.significant_count, self.moderate_count,
            self.minor_count, self.negligible_count,
            self.mean_score, self.max_score,
        )
    }
}

/// Convenience function to generate a report from scored conflicts.
pub fn generate_significance_report(conflicts: &[ScoredConflict]) -> SignificanceReport {
    SignificanceReport::generate(conflicts)
}

// ──────────────────────────── Tests ──────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Sex;
    use crate::Condition;
    

    fn test_patient() -> PatientProfile {
        PatientProfile::new(
            78.0, 70.0, Sex::Male,
        )
        .with_conditions(vec![
            Condition::new("I10", "Hypertension"),
            Condition::new("E11.9", "Type 2 Diabetes"),
            Condition::new("N18.3", "CKD stage 3"),
        ])
        .with_medications(vec![
            Medication::new("Warfarin", "anticoagulant", 5.0),
            Medication::new("Aspirin", "antiplatelet", 81.0),
            Medication::new("Lisinopril", "ACE inhibitor", 10.0),
            Medication::new("Metformin", "biguanide", 500.0),
            Medication::new("Simvastatin", "statin", 20.0),
        ])
        .with_egfr(45.0)
    }

    fn warfarin_aspirin_conflict() -> ConfirmedConflict {
        ConfirmedConflict::new(
            "Warfarin", "Aspirin", Severity::Major,
            "Additive anticoagulation", "Increased bleeding risk",
        )
    }

    fn simvastatin_amiodarone_conflict() -> ConfirmedConflict {
        ConfirmedConflict::new(
            "Simvastatin", "Amiodarone", Severity::Major,
            "CYP3A4 inhibition", "Rhabdomyolysis risk",
        )
    }

    fn minor_conflict() -> ConfirmedConflict {
        ConfirmedConflict::new(
            "Metformin", "Lisinopril", Severity::Minor,
            "Minor PD interaction", "Modest glucose-lowering enhancement",
        )
    }

    #[test]
    fn test_classify_severity() {
        assert_eq!(classify_severity(0.0), ClinicalSeverity::Negligible);
        assert_eq!(classify_severity(0.1), ClinicalSeverity::Negligible);
        assert_eq!(classify_severity(0.25), ClinicalSeverity::Minor);
        assert_eq!(classify_severity(0.45), ClinicalSeverity::Moderate);
        assert_eq!(classify_severity(0.65), ClinicalSeverity::Significant);
        assert_eq!(classify_severity(0.9), ClinicalSeverity::Critical);
    }

    #[test]
    fn test_severity_ordering() {
        assert!(ClinicalSeverity::Negligible < ClinicalSeverity::Minor);
        assert!(ClinicalSeverity::Minor < ClinicalSeverity::Moderate);
        assert!(ClinicalSeverity::Moderate < ClinicalSeverity::Significant);
        assert!(ClinicalSeverity::Significant < ClinicalSeverity::Critical);
    }

    #[test]
    fn test_base_severity_score() {
        assert!((base_severity_score(Severity::Minor) - 0.2).abs() < 1e-10);
        assert!((base_severity_score(Severity::Moderate) - 0.5).abs() < 1e-10);
        assert!((base_severity_score(Severity::Major) - 0.8).abs() < 1e-10);
        assert!((base_severity_score(Severity::Contraindicated) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_score_conflict_major() {
        let scorer = SignificanceScorer::with_defaults();
        let patient = test_patient();
        let conflict = warfarin_aspirin_conflict();
        let score = scorer.score_conflict(&conflict, &patient);

        assert!(score.total_score > 0.0, "Score should be positive");
        assert!(score.drugbank_component > 0.5, "DrugBank should rate this highly");
        assert!(
            score.severity_classification.at_least(ClinicalSeverity::Moderate),
            "Warfarin+aspirin should be at least moderate"
        );
    }

    #[test]
    fn test_score_conflict_minor() {
        let scorer = SignificanceScorer::with_defaults();
        let patient = test_patient();
        let conflict = minor_conflict();
        let score = scorer.score_conflict(&conflict, &patient);

        assert!(
            score.total_score < 0.6,
            "Minor conflict should score lower, got {}",
            score.total_score,
        );
    }

    #[test]
    fn test_major_scores_higher_than_minor() {
        let scorer = SignificanceScorer::with_defaults();
        let patient = test_patient();
        let major_score = scorer.score_conflict(&warfarin_aspirin_conflict(), &patient);
        let minor_score = scorer.score_conflict(&minor_conflict(), &patient);

        assert!(
            major_score.total_score > minor_score.total_score,
            "Major should score higher: {} vs {}",
            major_score.total_score, minor_score.total_score,
        );
    }

    #[test]
    fn test_score_all_conflicts() {
        let scorer = SignificanceScorer::with_defaults();
        let patient = test_patient();
        let conflicts = vec![
            warfarin_aspirin_conflict(),
            simvastatin_amiodarone_conflict(),
            minor_conflict(),
        ];
        let scored = scorer.score_all_conflicts(&conflicts, &patient);

        assert_eq!(scored.len(), 3);
        // Should be ranked by descending score
        assert_eq!(scored[0].rank, 1);
        assert_eq!(scored[1].rank, 2);
        assert_eq!(scored[2].rank, 3);
        assert!(scored[0].total_score() >= scored[1].total_score());
        assert!(scored[1].total_score() >= scored[2].total_score());
    }

    #[test]
    fn test_rank_conflicts() {
        let mut scored = vec![
            ScoredConflict {
                conflict: minor_conflict(),
                score: SignificanceScore { total_score: 0.3, ..SignificanceScore::zero() },
                rank: 0,
            },
            ScoredConflict {
                conflict: warfarin_aspirin_conflict(),
                score: SignificanceScore { total_score: 0.9, ..SignificanceScore::zero() },
                rank: 0,
            },
        ];
        rank_conflicts(&mut scored);
        assert_eq!(scored[0].rank, 1);
        assert!((scored[0].score.total_score - 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_filter_by_threshold() {
        let scored = vec![
            ScoredConflict {
                conflict: warfarin_aspirin_conflict(),
                score: SignificanceScore {
                    total_score: 0.85,
                    severity_classification: ClinicalSeverity::Critical,
                    ..SignificanceScore::zero()
                },
                rank: 1,
            },
            ScoredConflict {
                conflict: minor_conflict(),
                score: SignificanceScore {
                    total_score: 0.15,
                    severity_classification: ClinicalSeverity::Negligible,
                    ..SignificanceScore::zero()
                },
                rank: 2,
            },
        ];

        let filtered = filter_by_threshold(&scored, ClinicalSeverity::Moderate);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].severity(), ClinicalSeverity::Critical);
    }

    #[test]
    fn test_significance_report_empty() {
        let report = generate_significance_report(&[]);
        assert_eq!(report.total_conflicts, 0);
        assert_eq!(report.mean_score, 0.0);
    }

    #[test]
    fn test_significance_report() {
        let scorer = SignificanceScorer::with_defaults();
        let patient = test_patient();
        let conflicts = vec![
            warfarin_aspirin_conflict(),
            simvastatin_amiodarone_conflict(),
            minor_conflict(),
        ];
        let scored = scorer.score_all_conflicts(&conflicts, &patient);
        let report = generate_significance_report(&scored);

        assert_eq!(report.total_conflicts, 3);
        assert!(report.max_score >= report.min_score);
        assert!(report.mean_score > 0.0);
        assert!(report.top_conflicts.len() <= 5);
    }

    #[test]
    fn test_significance_score_display() {
        let score = SignificanceScore {
            total_score: 0.75,
            drugbank_component: 1.0,
            beers_component: 0.5,
            faers_component: 0.6,
            comorbidity_component: 0.4,
            severity_classification: ClinicalSeverity::Significant,
            base_severity_score: 0.8,
        };
        let s = format!("{}", score);
        assert!(s.contains("0.750"));
        assert!(s.contains("Significant"));
    }

    #[test]
    fn test_scored_conflict_display() {
        let sc = ScoredConflict {
            conflict: warfarin_aspirin_conflict(),
            score: SignificanceScore {
                total_score: 0.85,
                severity_classification: ClinicalSeverity::Critical,
                ..SignificanceScore::zero()
            },
            rank: 1,
        };
        let s = format!("{}", sc);
        assert!(s.contains("Warfarin"));
        assert!(s.contains("Aspirin"));
        assert!(s.contains("Critical"));
    }

    #[test]
    fn test_severity_at_least() {
        assert!(ClinicalSeverity::Critical.at_least(ClinicalSeverity::Negligible));
        assert!(ClinicalSeverity::Critical.at_least(ClinicalSeverity::Critical));
        assert!(!ClinicalSeverity::Minor.at_least(ClinicalSeverity::Moderate));
    }

    #[test]
    fn test_custom_weights() {
        let scorer = SignificanceScorer::new(1.0, 0.0, 0.0, 0.0);
        let patient = test_patient();
        let conflict = warfarin_aspirin_conflict();
        let score = scorer.score_conflict(&conflict, &patient);

        // With only DrugBank weight, score should equal drugbank component
        assert!((score.total_score - score.drugbank_component).abs() < 1e-10);
    }

    #[test]
    fn test_report_median() {
        let scored = vec![
            ScoredConflict {
                conflict: warfarin_aspirin_conflict(),
                score: SignificanceScore { total_score: 0.3, ..SignificanceScore::zero() },
                rank: 1,
            },
            ScoredConflict {
                conflict: minor_conflict(),
                score: SignificanceScore { total_score: 0.5, ..SignificanceScore::zero() },
                rank: 2,
            },
            ScoredConflict {
                conflict: simvastatin_amiodarone_conflict(),
                score: SignificanceScore { total_score: 0.7, ..SignificanceScore::zero() },
                rank: 3,
            },
        ];
        let report = generate_significance_report(&scored);
        assert!((report.median_score - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_significance_score_zero() {
        let score = SignificanceScore::zero();
        assert_eq!(score.total_score, 0.0);
        assert_eq!(score.severity_classification, ClinicalSeverity::Negligible);
    }

    #[test]
    fn test_score_non_elderly_beers_zero() {
        let scorer = SignificanceScorer::with_defaults();
        let patient = PatientProfile::new(
            40.0, 70.0, Sex::Male,
        );
        let conflict = warfarin_aspirin_conflict();
        let score = scorer.score_conflict(&conflict, &patient);
        assert_eq!(score.beers_component, 0.0, "Beers should not apply to non-elderly");
    }
}
