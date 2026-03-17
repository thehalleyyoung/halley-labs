//! TMR-style atemporal baseline interaction checker.
//!
//! Provides a simple table-based pairwise drug interaction database (the "dumb"
//! baseline) so that GuardPharma results can be compared against a traditional
//! atemporal approach.  This quantifies the improvement achieved by temporal
//! PK-aware formal verification over naïve lookup.

use std::collections::HashMap;
use std::fmt;

use serde::{Deserialize, Serialize};

use guardpharma_types::{DrugId, Severity};
use guardpharma_clinical::ActiveMedication;

// ═══════════════════════════════════════════════════════════════════════════
// Interaction Type
// ═══════════════════════════════════════════════════════════════════════════

/// Coarse interaction mechanism for the TMR baseline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InteractionType {
    PharmacoKinetic,
    PharmacoDynamic,
    Combined,
    Unknown,
}

impl fmt::Display for InteractionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PharmacoKinetic => write!(f, "PK"),
            Self::PharmacoDynamic => write!(f, "PD"),
            Self::Combined => write!(f, "PK+PD"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TMR Interaction
// ═══════════════════════════════════════════════════════════════════════════

/// A single pairwise interaction found by the TMR baseline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TmrInteraction {
    pub drug_a: DrugId,
    pub drug_b: DrugId,
    pub interaction_type: InteractionType,
    pub severity: Severity,
    pub description: String,
    pub evidence_source: String,
}

impl TmrInteraction {
    /// Canonical key for de-duplication (alphabetical order).
    pub fn pair_key(&self) -> (String, String) {
        let a = self.drug_a.as_str().to_string();
        let b = self.drug_b.as_str().to_string();
        if a <= b { (a, b) } else { (b, a) }
    }
}

impl fmt::Display for TmrInteraction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} ↔ {} [{}] {}",
            self.drug_a, self.drug_b, self.severity, self.description
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TMR Result
// ═══════════════════════════════════════════════════════════════════════════

/// Aggregated result of a TMR baseline check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TmrResult {
    pub interactions: Vec<TmrInteraction>,
    pub pairs_checked: usize,
    pub check_time_ms: f64,
}

impl TmrResult {
    pub fn new(interactions: Vec<TmrInteraction>, pairs_checked: usize) -> Self {
        Self { interactions, pairs_checked, check_time_ms: 0.0 }
    }

    pub fn with_time(mut self, ms: f64) -> Self {
        self.check_time_ms = ms;
        self
    }

    pub fn interaction_count(&self) -> usize {
        self.interactions.len()
    }

    /// Number of interactions at or above a given severity.
    pub fn count_at_severity(&self, min: Severity) -> usize {
        self.interactions.iter().filter(|i| i.severity >= min).count()
    }

    /// Unique drug pairs with interactions.
    pub fn unique_pairs(&self) -> Vec<(String, String)> {
        let mut pairs: Vec<(String, String)> = self.interactions.iter().map(|i| i.pair_key()).collect();
        pairs.sort();
        pairs.dedup();
        pairs
    }

    /// Maximum severity across all interactions.
    pub fn max_severity(&self) -> Option<Severity> {
        self.interactions.iter().map(|i| i.severity).max()
    }

    /// Filter interactions by minimum severity.
    pub fn filter_severity(&self, min: Severity) -> TmrResult {
        TmrResult {
            interactions: self.interactions.iter().filter(|i| i.severity >= min).cloned().collect(),
            pairs_checked: self.pairs_checked,
            check_time_ms: self.check_time_ms,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Database Entry
// ═══════════════════════════════════════════════════════════════════════════

/// A single entry in the TMR interaction database.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TmrDatabaseEntry {
    pub drug_a: String,
    pub drug_b: String,
    pub interaction_type: InteractionType,
    pub severity: Severity,
    pub description: String,
    pub evidence_source: String,
}

// ═══════════════════════════════════════════════════════════════════════════
// TMR Database
// ═══════════════════════════════════════════════════════════════════════════

/// Table-based drug interaction lookup database.
///
/// Uses a symmetric matrix keyed by canonical (alphabetically sorted) drug
/// name pairs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TmrDatabase {
    entries: HashMap<(String, String), TmrDatabaseEntry>,
    drug_count: usize,
}

impl TmrDatabase {
    /// Create an empty database.
    pub fn new() -> Self {
        Self { entries: HashMap::new(), drug_count: 0 }
    }

    /// Number of interaction entries.
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Canonical key (sorted pair).
    fn key(a: &str, b: &str) -> (String, String) {
        let al = a.to_lowercase().replace(' ', "_");
        let bl = b.to_lowercase().replace(' ', "_");
        if al <= bl { (al, bl) } else { (bl, al) }
    }

    /// Insert an interaction entry.
    pub fn insert(&mut self, entry: TmrDatabaseEntry) {
        let key = Self::key(&entry.drug_a, &entry.drug_b);
        self.entries.insert(key, entry);
    }

    /// Look up a pairwise interaction.
    pub fn lookup(&self, drug_a: &str, drug_b: &str) -> Option<&TmrDatabaseEntry> {
        let key = Self::key(drug_a, drug_b);
        self.entries.get(&key)
    }

    /// Check whether two drugs have a known interaction.
    pub fn has_interaction(&self, drug_a: &str, drug_b: &str) -> bool {
        self.lookup(drug_a, drug_b).is_some()
    }

    /// Set the expected number of unique drugs.
    pub fn set_drug_count(&mut self, n: usize) {
        self.drug_count = n;
    }

    /// All known drug names.
    pub fn known_drugs(&self) -> Vec<String> {
        let mut drugs: Vec<String> = self
            .entries
            .values()
            .flat_map(|e| vec![e.drug_a.to_lowercase(), e.drug_b.to_lowercase()])
            .collect();
        drugs.sort();
        drugs.dedup();
        drugs
    }
}

impl Default for TmrDatabase {
    fn default() -> Self {
        Self::new()
    }
}

/// Build the standard TMR interaction database with common drug interactions.
pub fn build_tmr_database() -> TmrDatabase {
    let mut db = TmrDatabase::new();

    // ── Warfarin interactions ───────────────────────────────────────────
    db.insert(TmrDatabaseEntry {
        drug_a: "warfarin".into(), drug_b: "aspirin".into(),
        interaction_type: InteractionType::PharmacoDynamic,
        severity: Severity::Major,
        description: "Additive anticoagulant/antiplatelet effect; increased bleeding risk".into(),
        evidence_source: "DrugBank / FDA label".into(),
    });
    db.insert(TmrDatabaseEntry {
        drug_a: "warfarin".into(), drug_b: "fluconazole".into(),
        interaction_type: InteractionType::PharmacoKinetic,
        severity: Severity::Major,
        description: "CYP2C9 inhibition increases warfarin exposure".into(),
        evidence_source: "DrugBank".into(),
    });
    db.insert(TmrDatabaseEntry {
        drug_a: "warfarin".into(), drug_b: "amiodarone".into(),
        interaction_type: InteractionType::Combined,
        severity: Severity::Contraindicated,
        description: "Amiodarone inhibits CYP2C9/CYP3A4 and has additive QT effects".into(),
        evidence_source: "FDA label".into(),
    });
    db.insert(TmrDatabaseEntry {
        drug_a: "warfarin".into(), drug_b: "rifampin".into(),
        interaction_type: InteractionType::PharmacoKinetic,
        severity: Severity::Major,
        description: "CYP2C9/3A4 induction substantially reduces warfarin levels".into(),
        evidence_source: "DrugBank".into(),
    });
    db.insert(TmrDatabaseEntry {
        drug_a: "warfarin".into(), drug_b: "ibuprofen".into(),
        interaction_type: InteractionType::PharmacoDynamic,
        severity: Severity::Major,
        description: "NSAID + anticoagulant: increased GI bleeding risk".into(),
        evidence_source: "DrugBank / Beers".into(),
    });

    // ── Statin interactions ─────────────────────────────────────────────
    db.insert(TmrDatabaseEntry {
        drug_a: "simvastatin".into(), drug_b: "amiodarone".into(),
        interaction_type: InteractionType::PharmacoKinetic,
        severity: Severity::Major,
        description: "CYP3A4 inhibition; rhabdomyolysis risk".into(),
        evidence_source: "FDA label".into(),
    });
    db.insert(TmrDatabaseEntry {
        drug_a: "simvastatin".into(), drug_b: "erythromycin".into(),
        interaction_type: InteractionType::PharmacoKinetic,
        severity: Severity::Contraindicated,
        description: "Strong CYP3A4 inhibition; high rhabdomyolysis risk".into(),
        evidence_source: "FDA label".into(),
    });
    db.insert(TmrDatabaseEntry {
        drug_a: "simvastatin".into(), drug_b: "clarithromycin".into(),
        interaction_type: InteractionType::PharmacoKinetic,
        severity: Severity::Contraindicated,
        description: "Strong CYP3A4 inhibition; high rhabdomyolysis risk".into(),
        evidence_source: "FDA label".into(),
    });
    db.insert(TmrDatabaseEntry {
        drug_a: "simvastatin".into(), drug_b: "diltiazem".into(),
        interaction_type: InteractionType::PharmacoKinetic,
        severity: Severity::Major,
        description: "Moderate CYP3A4 inhibition; limit simvastatin to 10mg".into(),
        evidence_source: "FDA label".into(),
    });
    db.insert(TmrDatabaseEntry {
        drug_a: "atorvastatin".into(), drug_b: "clarithromycin".into(),
        interaction_type: InteractionType::PharmacoKinetic,
        severity: Severity::Major,
        description: "CYP3A4 inhibition increases statin exposure".into(),
        evidence_source: "DrugBank".into(),
    });

    // ── Metformin interactions ──────────────────────────────────────────
    db.insert(TmrDatabaseEntry {
        drug_a: "metformin".into(), drug_b: "alcohol".into(),
        interaction_type: InteractionType::PharmacoDynamic,
        severity: Severity::Major,
        description: "Increased lactic acidosis risk".into(),
        evidence_source: "FDA label".into(),
    });
    db.insert(TmrDatabaseEntry {
        drug_a: "metformin".into(), drug_b: "contrast_dye".into(),
        interaction_type: InteractionType::PharmacoKinetic,
        severity: Severity::Major,
        description: "Iodinated contrast + metformin: lactic acidosis risk in renal impairment".into(),
        evidence_source: "ACR guidelines".into(),
    });

    // ── SSRI interactions ───────────────────────────────────────────────
    db.insert(TmrDatabaseEntry {
        drug_a: "fluoxetine".into(), drug_b: "tramadol".into(),
        interaction_type: InteractionType::Combined,
        severity: Severity::Major,
        description: "Serotonin syndrome risk; CYP2D6 inhibition reduces tramadol efficacy".into(),
        evidence_source: "DrugBank / FDA".into(),
    });
    db.insert(TmrDatabaseEntry {
        drug_a: "fluoxetine".into(), drug_b: "linezolid".into(),
        interaction_type: InteractionType::PharmacoDynamic,
        severity: Severity::Contraindicated,
        description: "MAO inhibition + SSRI: severe serotonin syndrome risk".into(),
        evidence_source: "FDA label".into(),
    });
    db.insert(TmrDatabaseEntry {
        drug_a: "sertraline".into(), drug_b: "tramadol".into(),
        interaction_type: InteractionType::PharmacoDynamic,
        severity: Severity::Major,
        description: "Serotonin syndrome risk".into(),
        evidence_source: "DrugBank".into(),
    });

    // ── Cardiac interactions ────────────────────────────────────────────
    db.insert(TmrDatabaseEntry {
        drug_a: "digoxin".into(), drug_b: "amiodarone".into(),
        interaction_type: InteractionType::PharmacoKinetic,
        severity: Severity::Major,
        description: "Amiodarone increases digoxin levels; reduce digoxin dose by 50%".into(),
        evidence_source: "FDA label".into(),
    });
    db.insert(TmrDatabaseEntry {
        drug_a: "digoxin".into(), drug_b: "verapamil".into(),
        interaction_type: InteractionType::PharmacoKinetic,
        severity: Severity::Major,
        description: "P-gp inhibition increases digoxin exposure".into(),
        evidence_source: "DrugBank".into(),
    });
    db.insert(TmrDatabaseEntry {
        drug_a: "metoprolol".into(), drug_b: "verapamil".into(),
        interaction_type: InteractionType::PharmacoDynamic,
        severity: Severity::Major,
        description: "Additive negative chronotropic/inotropic effects; risk of heart block".into(),
        evidence_source: "FDA label".into(),
    });
    db.insert(TmrDatabaseEntry {
        drug_a: "amiodarone".into(), drug_b: "sotalol".into(),
        interaction_type: InteractionType::PharmacoDynamic,
        severity: Severity::Contraindicated,
        description: "Additive QT prolongation; torsades de pointes risk".into(),
        evidence_source: "FDA label".into(),
    });

    // ── Immunosuppressant interactions ──────────────────────────────────
    db.insert(TmrDatabaseEntry {
        drug_a: "cyclosporine".into(), drug_b: "ketoconazole".into(),
        interaction_type: InteractionType::PharmacoKinetic,
        severity: Severity::Major,
        description: "CYP3A4 inhibition markedly increases cyclosporine levels".into(),
        evidence_source: "DrugBank".into(),
    });
    db.insert(TmrDatabaseEntry {
        drug_a: "tacrolimus".into(), drug_b: "fluconazole".into(),
        interaction_type: InteractionType::PharmacoKinetic,
        severity: Severity::Major,
        description: "CYP3A4 inhibition increases tacrolimus exposure; nephrotoxicity risk".into(),
        evidence_source: "DrugBank".into(),
    });
    db.insert(TmrDatabaseEntry {
        drug_a: "tacrolimus".into(), drug_b: "erythromycin".into(),
        interaction_type: InteractionType::PharmacoKinetic,
        severity: Severity::Major,
        description: "CYP3A4 inhibition increases tacrolimus levels".into(),
        evidence_source: "DrugBank".into(),
    });

    // ── Opioid interactions ─────────────────────────────────────────────
    db.insert(TmrDatabaseEntry {
        drug_a: "oxycodone".into(), drug_b: "benzodiazepine".into(),
        interaction_type: InteractionType::PharmacoDynamic,
        severity: Severity::Contraindicated,
        description: "CNS/respiratory depression; FDA boxed warning".into(),
        evidence_source: "FDA boxed warning".into(),
    });
    db.insert(TmrDatabaseEntry {
        drug_a: "morphine".into(), drug_b: "benzodiazepine".into(),
        interaction_type: InteractionType::PharmacoDynamic,
        severity: Severity::Contraindicated,
        description: "CNS/respiratory depression; FDA boxed warning".into(),
        evidence_source: "FDA boxed warning".into(),
    });

    // ── Renal interactions ──────────────────────────────────────────────
    db.insert(TmrDatabaseEntry {
        drug_a: "lisinopril".into(), drug_b: "spironolactone".into(),
        interaction_type: InteractionType::PharmacoDynamic,
        severity: Severity::Major,
        description: "Hyperkalemia risk: ACE inhibitor + potassium-sparing diuretic".into(),
        evidence_source: "DrugBank".into(),
    });
    db.insert(TmrDatabaseEntry {
        drug_a: "lisinopril".into(), drug_b: "potassium_supplement".into(),
        interaction_type: InteractionType::PharmacoDynamic,
        severity: Severity::Major,
        description: "Hyperkalemia risk with ACE inhibitor + potassium supplementation".into(),
        evidence_source: "Clinical guideline".into(),
    });

    // ── QT prolongation pairs ───────────────────────────────────────────
    db.insert(TmrDatabaseEntry {
        drug_a: "ciprofloxacin".into(), drug_b: "amiodarone".into(),
        interaction_type: InteractionType::PharmacoDynamic,
        severity: Severity::Major,
        description: "Additive QT prolongation".into(),
        evidence_source: "CredibleMeds".into(),
    });
    db.insert(TmrDatabaseEntry {
        drug_a: "haloperidol".into(), drug_b: "methadone".into(),
        interaction_type: InteractionType::PharmacoDynamic,
        severity: Severity::Major,
        description: "Additive QT prolongation risk".into(),
        evidence_source: "CredibleMeds".into(),
    });

    // ── Moderate interactions ───────────────────────────────────────────
    db.insert(TmrDatabaseEntry {
        drug_a: "amlodipine".into(), drug_b: "simvastatin".into(),
        interaction_type: InteractionType::PharmacoKinetic,
        severity: Severity::Moderate,
        description: "Amlodipine weakly inhibits CYP3A4; limit simvastatin to 20mg".into(),
        evidence_source: "FDA label".into(),
    });
    db.insert(TmrDatabaseEntry {
        drug_a: "omeprazole".into(), drug_b: "clopidogrel".into(),
        interaction_type: InteractionType::PharmacoKinetic,
        severity: Severity::Moderate,
        description: "CYP2C19 inhibition reduces clopidogrel activation".into(),
        evidence_source: "FDA label".into(),
    });
    db.insert(TmrDatabaseEntry {
        drug_a: "metformin".into(), drug_b: "furosemide".into(),
        interaction_type: InteractionType::PharmacoKinetic,
        severity: Severity::Moderate,
        description: "Furosemide increases metformin Cmax; monitor renal function".into(),
        evidence_source: "DrugBank".into(),
    });

    // ── Minor interactions ──────────────────────────────────────────────
    db.insert(TmrDatabaseEntry {
        drug_a: "levothyroxine".into(), drug_b: "calcium".into(),
        interaction_type: InteractionType::PharmacoKinetic,
        severity: Severity::Minor,
        description: "Calcium reduces levothyroxine absorption; separate by 4 hours".into(),
        evidence_source: "Clinical practice".into(),
    });
    db.insert(TmrDatabaseEntry {
        drug_a: "levothyroxine".into(), drug_b: "iron".into(),
        interaction_type: InteractionType::PharmacoKinetic,
        severity: Severity::Minor,
        description: "Iron reduces levothyroxine absorption; separate by 4 hours".into(),
        evidence_source: "Clinical practice".into(),
    });

    db.set_drug_count(40);
    db
}

// ═══════════════════════════════════════════════════════════════════════════
// TMR Baseline Checker
// ═══════════════════════════════════════════════════════════════════════════

/// TMR-style atemporal baseline interaction checker.
///
/// Simply iterates over all pairwise drug combinations and looks each up in a
/// flat interaction database — no temporal reasoning, no PK modelling.
#[derive(Debug, Clone)]
pub struct TmrBaseline {
    pub database: TmrDatabase,
}

impl TmrBaseline {
    /// Create with the standard built-in database.
    pub fn new() -> Self {
        Self { database: build_tmr_database() }
    }

    /// Create with a custom database.
    pub fn with_database(database: TmrDatabase) -> Self {
        Self { database }
    }

    /// Check all pairwise interactions among a medication list.
    pub fn check_interactions(&self, medications: &[ActiveMedication]) -> TmrResult {
        let start = std::time::Instant::now();
        let mut interactions = Vec::new();
        let mut pairs_checked = 0usize;

        for i in 0..medications.len() {
            for j in (i + 1)..medications.len() {
                pairs_checked += 1;
                let a = medications[i].drug_id.as_str();
                let b = medications[j].drug_id.as_str();
                if let Some(entry) = self.database.lookup(a, b) {
                    interactions.push(TmrInteraction {
                        drug_a: medications[i].drug_id.clone(),
                        drug_b: medications[j].drug_id.clone(),
                        interaction_type: entry.interaction_type,
                        severity: entry.severity,
                        description: entry.description.clone(),
                        evidence_source: entry.evidence_source.clone(),
                    });
                }
            }
        }

        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        TmrResult::new(interactions, pairs_checked).with_time(elapsed)
    }

    /// Check interactions by drug IDs only (without full ActiveMedication data).
    pub fn check_by_ids(&self, drug_ids: &[DrugId]) -> TmrResult {
        let start = std::time::Instant::now();
        let mut interactions = Vec::new();
        let mut pairs_checked = 0usize;

        for i in 0..drug_ids.len() {
            for j in (i + 1)..drug_ids.len() {
                pairs_checked += 1;
                let a = drug_ids[i].as_str();
                let b = drug_ids[j].as_str();
                if let Some(entry) = self.database.lookup(a, b) {
                    interactions.push(TmrInteraction {
                        drug_a: drug_ids[i].clone(),
                        drug_b: drug_ids[j].clone(),
                        interaction_type: entry.interaction_type,
                        severity: entry.severity,
                        description: entry.description.clone(),
                        evidence_source: entry.evidence_source.clone(),
                    });
                }
            }
        }

        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        TmrResult::new(interactions, pairs_checked).with_time(elapsed)
    }

    /// Interaction density (fraction of checked pairs that have interactions).
    pub fn interaction_density(&self, medications: &[ActiveMedication]) -> f64 {
        let result = self.check_interactions(medications);
        if result.pairs_checked == 0 { 0.0 } else { result.interaction_count() as f64 / result.pairs_checked as f64 }
    }
}

impl Default for TmrBaseline {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Verification Result (local representation for comparison)
// ═══════════════════════════════════════════════════════════════════════════

/// Simplified verification result for comparison purposes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    pub drug_a: DrugId,
    pub drug_b: DrugId,
    pub conflict_detected: bool,
    pub severity: Option<Severity>,
    pub temporal: bool,
    pub tier: u8,
    pub description: String,
}

impl VerificationResult {
    pub fn conflict(drug_a: DrugId, drug_b: DrugId, severity: Severity, temporal: bool, tier: u8) -> Self {
        Self {
            drug_a, drug_b, conflict_detected: true,
            severity: Some(severity), temporal, tier,
            description: String::new(),
        }
    }

    pub fn safe(drug_a: DrugId, drug_b: DrugId) -> Self {
        Self {
            drug_a, drug_b, conflict_detected: false,
            severity: None, temporal: false, tier: 1,
            description: String::new(),
        }
    }

    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = desc.to_string();
        self
    }

    pub fn pair_key(&self) -> (String, String) {
        let a = self.drug_a.as_str().to_string();
        let b = self.drug_b.as_str().to_string();
        if a <= b { (a, b) } else { (b, a) }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Baseline Comparison
// ═══════════════════════════════════════════════════════════════════════════

/// Comparison of GuardPharma results against the TMR baseline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineComparison {
    /// Conflicts found by both GuardPharma and TMR.
    pub true_positives: usize,
    /// Conflicts found by GuardPharma but missed by TMR (temporal / novel).
    pub false_negatives_baseline: usize,
    /// Conflicts found by TMR but ruled out by GuardPharma (false positives).
    pub false_positives_baseline: usize,
    /// Pairs marked safe by both.
    pub true_negatives: usize,
    /// Total pairs analysed.
    pub total_pairs: usize,
    /// Detailed pair results.
    pub pair_details: Vec<PairComparisonDetail>,
}

/// Detail for one drug pair in the comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairComparisonDetail {
    pub drug_a: String,
    pub drug_b: String,
    pub tmr_found: bool,
    pub guardpharma_found: bool,
    pub tmr_severity: Option<Severity>,
    pub guardpharma_severity: Option<Severity>,
    pub temporal: bool,
}

impl BaselineComparison {
    /// Accuracy of the baseline relative to GuardPharma (ground truth).
    pub fn baseline_accuracy(&self) -> f64 {
        let t = self.total_pairs;
        if t == 0 { return 0.0; }
        (self.true_positives + self.true_negatives) as f64 / t as f64
    }

    /// Baseline sensitivity (recall): among GuardPharma positives, how many TMR found.
    pub fn baseline_sensitivity(&self) -> f64 {
        let total_positives = self.true_positives + self.false_negatives_baseline;
        if total_positives == 0 { 0.0 } else { self.true_positives as f64 / total_positives as f64 }
    }

    /// Baseline specificity.
    pub fn baseline_specificity(&self) -> f64 {
        let total_negatives = self.true_negatives + self.false_positives_baseline;
        if total_negatives == 0 { 0.0 } else { self.true_negatives as f64 / total_negatives as f64 }
    }

    /// Number of temporal conflicts missed by baseline.
    pub fn temporal_misses(&self) -> usize {
        self.pair_details.iter().filter(|d| d.temporal && d.guardpharma_found && !d.tmr_found).count()
    }

    /// Severity agreement rate for pairs found by both.
    pub fn severity_agreement_rate(&self) -> f64 {
        let both_found: Vec<&PairComparisonDetail> = self
            .pair_details
            .iter()
            .filter(|d| d.tmr_found && d.guardpharma_found)
            .collect();
        if both_found.is_empty() { return 1.0; }
        let agreed = both_found
            .iter()
            .filter(|d| d.tmr_severity == d.guardpharma_severity)
            .count();
        agreed as f64 / both_found.len() as f64
    }
}

impl fmt::Display for BaselineComparison {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Baseline comparison: TP={}, FN_base={}, FP_base={}, TN={} (acc={:.3})",
            self.true_positives,
            self.false_negatives_baseline,
            self.false_positives_baseline,
            self.true_negatives,
            self.baseline_accuracy(),
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Improvement Metrics
// ═══════════════════════════════════════════════════════════════════════════

/// Quantified improvement of GuardPharma over the TMR baseline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImprovementMetrics {
    /// Conflicts GuardPharma found that TMR missed.
    pub additional_conflicts_found: usize,
    /// False positives from TMR that GuardPharma correctly rejected.
    pub false_positives_eliminated: usize,
    /// Temporal-only conflicts (unique to temporal analysis).
    pub temporal_conflicts_count: usize,
    /// Severity corrections (where GuardPharma adjusted severity).
    pub severity_corrections: usize,
    /// Relative improvement in sensitivity.
    pub sensitivity_improvement: f64,
    /// Relative improvement in specificity.
    pub specificity_improvement: f64,
    /// Time overhead ratio (GuardPharma time / TMR time).
    pub time_overhead_ratio: f64,
}

impl ImprovementMetrics {
    /// Overall improvement score (0-1 scale).
    pub fn overall_score(&self) -> f64 {
        let conflict_score = if self.additional_conflicts_found > 0 { 0.3 } else { 0.0 };
        let fp_score = if self.false_positives_eliminated > 0 { 0.2 } else { 0.0 };
        let temporal_score = if self.temporal_conflicts_count > 0 { 0.3 } else { 0.0 };
        let correction_score = if self.severity_corrections > 0 { 0.2 } else { 0.0 };
        conflict_score + fp_score + temporal_score + correction_score
    }

    /// Whether any meaningful improvement was observed.
    pub fn has_improvement(&self) -> bool {
        self.additional_conflicts_found > 0
            || self.false_positives_eliminated > 0
            || self.temporal_conflicts_count > 0
    }
}

impl fmt::Display for ImprovementMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "+{} conflicts, -{} FP, {} temporal, {} corrections (score={:.2})",
            self.additional_conflicts_found,
            self.false_positives_eliminated,
            self.temporal_conflicts_count,
            self.severity_corrections,
            self.overall_score(),
        )
    }
}

/// Compute baseline comparison between GuardPharma results and TMR baseline.
pub fn compare_results(
    guardpharma: &[VerificationResult],
    tmr: &TmrResult,
    total_pairs: usize,
) -> BaselineComparison {
    let gp_map: HashMap<(String, String), &VerificationResult> =
        guardpharma.iter().filter(|r| r.conflict_detected).map(|r| (r.pair_key(), r)).collect();
    let tmr_map: HashMap<(String, String), &TmrInteraction> =
        tmr.interactions.iter().map(|i| (i.pair_key(), i)).collect();

    let mut all_keys: Vec<(String, String)> = gp_map.keys().chain(tmr_map.keys()).cloned().collect();
    all_keys.sort();
    all_keys.dedup();

    let mut tp = 0usize;
    let mut fn_base = 0usize;
    let mut fp_base = 0usize;
    let mut details = Vec::new();

    for key in &all_keys {
        let gp = gp_map.get(key);
        let tm = tmr_map.get(key);
        let gp_found = gp.is_some();
        let tmr_found = tm.is_some();

        match (gp_found, tmr_found) {
            (true, true) => tp += 1,
            (true, false) => fn_base += 1,
            (false, true) => fp_base += 1,
            (false, false) => {}
        }

        details.push(PairComparisonDetail {
            drug_a: key.0.clone(),
            drug_b: key.1.clone(),
            tmr_found,
            guardpharma_found: gp_found,
            tmr_severity: tm.map(|t| t.severity),
            guardpharma_severity: gp.and_then(|g| g.severity),
            temporal: gp.map(|g| g.temporal).unwrap_or(false),
        });
    }

    let classified = tp + fn_base + fp_base;
    let tn = if total_pairs > classified { total_pairs - classified } else { 0 };

    BaselineComparison {
        true_positives: tp,
        false_negatives_baseline: fn_base,
        false_positives_baseline: fp_base,
        true_negatives: tn,
        total_pairs,
        pair_details: details,
    }
}

/// Compute improvement metrics from a baseline comparison.
pub fn compute_improvement(
    guardpharma: &[VerificationResult],
    baseline: &TmrResult,
    gp_time_ms: f64,
) -> ImprovementMetrics {
    let gp_conflicts: HashMap<(String, String), &VerificationResult> =
        guardpharma.iter().filter(|r| r.conflict_detected).map(|r| (r.pair_key(), r)).collect();
    let tmr_set: HashMap<(String, String), &TmrInteraction> =
        baseline.interactions.iter().map(|i| (i.pair_key(), i)).collect();

    let additional = gp_conflicts.keys().filter(|k| !tmr_set.contains_key(*k)).count();
    let eliminated = tmr_set.keys().filter(|k| !gp_conflicts.contains_key(*k)).count();
    let temporal = gp_conflicts.values().filter(|v| v.temporal).count();

    let mut severity_corrections = 0usize;
    for (key, gp) in &gp_conflicts {
        if let Some(tmr) = tmr_set.get(key) {
            if gp.severity != Some(tmr.severity) {
                severity_corrections += 1;
            }
        }
    }

    let total_gp_positive = gp_conflicts.len();
    let total_tmr_positive = tmr_set.len();
    let gp_sens = if total_gp_positive > 0 { 1.0 } else { 0.0 };
    let tmr_sens = if total_gp_positive > 0 {
        gp_conflicts.keys().filter(|k| tmr_set.contains_key(*k)).count() as f64 / total_gp_positive as f64
    } else {
        0.0
    };

    let time_overhead = if baseline.check_time_ms > 0.0 {
        gp_time_ms / baseline.check_time_ms
    } else {
        1.0
    };

    ImprovementMetrics {
        additional_conflicts_found: additional,
        false_positives_eliminated: eliminated,
        temporal_conflicts_count: temporal,
        severity_corrections,
        sensitivity_improvement: gp_sens - tmr_sens,
        specificity_improvement: if total_tmr_positive > 0 { eliminated as f64 / total_tmr_positive as f64 } else { 0.0 },
        time_overhead_ratio: time_overhead,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use guardpharma_types::{DrugInfo, DosingSchedule};

    fn make_med(name: &str) -> ActiveMedication {
        ActiveMedication::new(
            DrugId::new(name),
            DrugInfo::new(name, "Test"),
            DosingSchedule::new(10.0, 24.0),
        )
    }

    #[test]
    fn test_build_tmr_database() {
        let db = build_tmr_database();
        assert!(db.entry_count() > 20);
        assert!(db.has_interaction("warfarin", "aspirin"));
        assert!(db.has_interaction("aspirin", "warfarin")); // symmetric
    }

    #[test]
    fn test_tmr_database_lookup() {
        let db = build_tmr_database();
        let entry = db.lookup("warfarin", "fluconazole").unwrap();
        assert_eq!(entry.severity, Severity::Major);
        assert_eq!(entry.interaction_type, InteractionType::PharmacoKinetic);
    }

    #[test]
    fn test_tmr_database_unknown_pair() {
        let db = build_tmr_database();
        assert!(db.lookup("warfarin", "vitamin_c").is_none());
    }

    #[test]
    fn test_tmr_baseline_check() {
        let baseline = TmrBaseline::new();
        let meds = vec![make_med("warfarin"), make_med("aspirin"), make_med("metformin")];
        let result = baseline.check_interactions(&meds);
        assert_eq!(result.pairs_checked, 3);
        assert!(result.interaction_count() >= 1);
        // warfarin+aspirin should be found
        assert!(result.interactions.iter().any(|i| {
            let (a, b) = i.pair_key();
            (a == "aspirin" && b == "warfarin") || (a == "warfarin" && b == "aspirin")
        }));
    }

    #[test]
    fn test_tmr_baseline_no_interactions() {
        let baseline = TmrBaseline::new();
        let meds = vec![make_med("vitamin_d"), make_med("melatonin")];
        let result = baseline.check_interactions(&meds);
        assert_eq!(result.interaction_count(), 0);
    }

    #[test]
    fn test_tmr_result_filter_severity() {
        let baseline = TmrBaseline::new();
        let meds = vec![
            make_med("warfarin"), make_med("aspirin"), make_med("amiodarone"),
        ];
        let result = baseline.check_interactions(&meds);
        let major = result.filter_severity(Severity::Major);
        assert!(major.interaction_count() <= result.interaction_count());
        let contra = result.filter_severity(Severity::Contraindicated);
        assert!(contra.interaction_count() <= major.interaction_count());
    }

    #[test]
    fn test_check_by_ids() {
        let baseline = TmrBaseline::new();
        let ids = vec![DrugId::new("warfarin"), DrugId::new("fluconazole")];
        let result = baseline.check_by_ids(&ids);
        assert_eq!(result.pairs_checked, 1);
        assert_eq!(result.interaction_count(), 1);
    }

    #[test]
    fn test_verification_result_pair_key() {
        let r = VerificationResult::conflict(
            DrugId::new("warfarin"), DrugId::new("aspirin"),
            Severity::Major, false, 1,
        );
        let (a, b) = r.pair_key();
        assert!(a <= b);
    }

    #[test]
    fn test_compare_results() {
        let tmr = TmrBaseline::new();
        let meds = vec![make_med("warfarin"), make_med("aspirin")];
        let tmr_result = tmr.check_interactions(&meds);

        let gp_results = vec![
            VerificationResult::conflict(
                DrugId::new("warfarin"), DrugId::new("aspirin"),
                Severity::Major, true, 2,
            ),
        ];

        let comparison = compare_results(&gp_results, &tmr_result, 1);
        assert_eq!(comparison.true_positives, 1);
        assert_eq!(comparison.false_negatives_baseline, 0);
    }

    #[test]
    fn test_improvement_metrics_with_temporal() {
        let tmr_result = TmrResult::new(vec![], 3).with_time(1.0);
        let gp_results = vec![
            VerificationResult::conflict(
                DrugId::new("drug_a"), DrugId::new("drug_b"),
                Severity::Major, true, 2,
            ),
        ];

        let improvement = compute_improvement(&gp_results, &tmr_result, 10.0);
        assert_eq!(improvement.additional_conflicts_found, 1);
        assert_eq!(improvement.temporal_conflicts_count, 1);
        assert!(improvement.has_improvement());
    }

    #[test]
    fn test_improvement_metrics_eliminated_fp() {
        let tmr_result = TmrResult::new(
            vec![TmrInteraction {
                drug_a: DrugId::new("x"), drug_b: DrugId::new("y"),
                interaction_type: InteractionType::Unknown,
                severity: Severity::Moderate,
                description: "Suspected".into(),
                evidence_source: "test".into(),
            }],
            1,
        ).with_time(1.0);

        let gp_results: Vec<VerificationResult> = vec![];
        let improvement = compute_improvement(&gp_results, &tmr_result, 5.0);
        assert_eq!(improvement.false_positives_eliminated, 1);
    }

    #[test]
    fn test_baseline_comparison_metrics() {
        let comparison = BaselineComparison {
            true_positives: 8,
            false_negatives_baseline: 3,
            false_positives_baseline: 2,
            true_negatives: 37,
            total_pairs: 50,
            pair_details: vec![],
        };
        assert!((comparison.baseline_accuracy() - 0.9).abs() < 1e-10);
        assert!((comparison.baseline_sensitivity() - 8.0 / 11.0).abs() < 1e-10);
    }

    #[test]
    fn test_tmr_known_drugs() {
        let db = build_tmr_database();
        let drugs = db.known_drugs();
        assert!(drugs.contains(&"warfarin".to_string()));
        assert!(drugs.contains(&"aspirin".to_string()));
    }

    #[test]
    fn test_interaction_density() {
        let baseline = TmrBaseline::new();
        let meds = vec![make_med("warfarin"), make_med("aspirin")];
        let density = baseline.interaction_density(&meds);
        assert!((density - 1.0).abs() < 1e-10); // 1 interaction / 1 pair
    }
}
