//! DrugBank interaction severity database.
//!
//! Provides lookup of known drug–drug interaction severity, mechanism, and
//! evidence level based on DrugBank reference data. The built-in database
//! contains ~120 of the most clinically important interactions.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ─────────────────────────── Enums ───────────────────────────────────────

/// DrugBank severity classification for a drug–drug interaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub enum DrugBankSeverity {
    Unknown,
    Minor,
    Moderate,
    Major,
}

impl DrugBankSeverity {
    /// Numeric score in [0, 1] for weighted combination.
    pub fn score(&self) -> f64 {
        match self {
            DrugBankSeverity::Unknown => 0.0,
            DrugBankSeverity::Minor => 0.25,
            DrugBankSeverity::Moderate => 0.55,
            DrugBankSeverity::Major => 1.0,
        }
    }

    pub fn is_major(&self) -> bool {
        matches!(self, DrugBankSeverity::Major)
    }

    pub fn is_at_least_moderate(&self) -> bool {
        matches!(self, DrugBankSeverity::Moderate | DrugBankSeverity::Major)
    }
}

impl fmt::Display for DrugBankSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DrugBankSeverity::Unknown => write!(f, "Unknown"),
            DrugBankSeverity::Minor => write!(f, "Minor"),
            DrugBankSeverity::Moderate => write!(f, "Moderate"),
            DrugBankSeverity::Major => write!(f, "Major"),
        }
    }
}

impl Default for DrugBankSeverity {
    fn default() -> Self {
        DrugBankSeverity::Unknown
    }
}

/// Level of evidence supporting an interaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub enum EvidenceLevel {
    Unlikely,
    Possible,
    Suspected,
    Probable,
    Established,
}

impl EvidenceLevel {
    /// Numeric weight in [0, 1].
    pub fn weight(&self) -> f64 {
        match self {
            EvidenceLevel::Unlikely => 0.1,
            EvidenceLevel::Possible => 0.3,
            EvidenceLevel::Suspected => 0.5,
            EvidenceLevel::Probable => 0.75,
            EvidenceLevel::Established => 1.0,
        }
    }
}

impl fmt::Display for EvidenceLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EvidenceLevel::Unlikely => write!(f, "Unlikely"),
            EvidenceLevel::Possible => write!(f, "Possible"),
            EvidenceLevel::Suspected => write!(f, "Suspected"),
            EvidenceLevel::Probable => write!(f, "Probable"),
            EvidenceLevel::Established => write!(f, "Established"),
        }
    }
}

impl Default for EvidenceLevel {
    fn default() -> Self {
        EvidenceLevel::Possible
    }
}

// ─────────────────────────── Structs ─────────────────────────────────────

/// A single drug–drug interaction record from DrugBank.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrugBankInteraction {
    pub drug_a: String,
    pub drug_b: String,
    pub severity: DrugBankSeverity,
    pub description: String,
    pub mechanism: String,
    pub evidence_level: EvidenceLevel,
    pub clinical_consequence: String,
    pub management: String,
}

impl DrugBankInteraction {
    pub fn new(
        drug_a: &str,
        drug_b: &str,
        severity: DrugBankSeverity,
        description: &str,
        mechanism: &str,
        evidence_level: EvidenceLevel,
    ) -> Self {
        DrugBankInteraction {
            drug_a: drug_a.to_lowercase(),
            drug_b: drug_b.to_lowercase(),
            severity,
            description: description.to_string(),
            mechanism: mechanism.to_string(),
            evidence_level,
            clinical_consequence: String::new(),
            management: String::new(),
        }
    }

    pub fn with_consequence(mut self, consequence: &str) -> Self {
        self.clinical_consequence = consequence.to_string();
        self
    }

    pub fn with_management(mut self, management: &str) -> Self {
        self.management = management.to_string();
        self
    }

    /// Composite score = severity × evidence weight.
    pub fn composite_score(&self) -> f64 {
        self.severity.score() * self.evidence_level.weight()
    }

    /// Canonical pair key (alphabetical order).
    pub fn pair_key(&self) -> (String, String) {
        let a = self.drug_a.clone();
        let b = self.drug_b.clone();
        if a <= b { (a, b) } else { (b, a) }
    }

    pub fn involves(&self, drug: &str) -> bool {
        let d = drug.to_lowercase();
        self.drug_a == d || self.drug_b == d
    }

    pub fn involves_pair(&self, drug_a: &str, drug_b: &str) -> bool {
        let a = drug_a.to_lowercase();
        let b = drug_b.to_lowercase();
        (self.drug_a == a && self.drug_b == b) || (self.drug_a == b && self.drug_b == a)
    }
}

impl fmt::Display for DrugBankInteraction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} ↔ {} [{}] ({}): {}",
            self.drug_a, self.drug_b, self.severity, self.evidence_level, self.description
        )
    }
}

/// A DrugBank entry for a single drug, listing all its interactions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrugBankEntry {
    pub drug_id: String,
    pub name: String,
    pub interactions: Vec<DrugBankInteraction>,
}

impl DrugBankEntry {
    pub fn new(drug_id: &str, name: &str) -> Self {
        DrugBankEntry {
            drug_id: drug_id.to_string(),
            name: name.to_string(),
            interactions: Vec::new(),
        }
    }

    pub fn add_interaction(&mut self, interaction: DrugBankInteraction) {
        self.interactions.push(interaction);
    }

    pub fn major_interactions(&self) -> Vec<&DrugBankInteraction> {
        self.interactions.iter().filter(|i| i.severity.is_major()).collect()
    }

    pub fn interaction_count(&self) -> usize {
        self.interactions.len()
    }
}

// ─────────────────────────── Database ────────────────────────────────────

/// In-memory DrugBank interaction database.
///
/// Stores interactions keyed by canonical pair (alphabetically ordered, lowered).
#[derive(Debug, Clone)]
pub struct DrugBankDatabase {
    interactions: HashMap<(String, String), DrugBankInteraction>,
    drug_index: HashMap<String, Vec<(String, String)>>,
}

impl DrugBankDatabase {
    /// Create an empty database.
    pub fn empty() -> Self {
        DrugBankDatabase {
            interactions: HashMap::new(),
            drug_index: HashMap::new(),
        }
    }

    /// Create the database pre-populated with common interactions.
    pub fn with_defaults() -> Self {
        let mut db = Self::empty();
        db.load_defaults();
        db
    }

    /// Insert an interaction into the database.
    pub fn insert(&mut self, interaction: DrugBankInteraction) {
        let key = interaction.pair_key();
        let a = key.0.clone();
        let b = key.1.clone();
        self.drug_index.entry(a.clone()).or_default().push(key.clone());
        self.drug_index.entry(b.clone()).or_default().push(key.clone());
        self.interactions.insert(key, interaction);
    }

    /// Lookup a specific drug pair.
    pub fn get_interaction(&self, drug_a: &str, drug_b: &str) -> Option<&DrugBankInteraction> {
        let key = canonical_key(drug_a, drug_b);
        self.interactions.get(&key)
    }

    /// Get severity for a drug pair.
    pub fn get_interaction_severity(
        &self,
        drug_a: &str,
        drug_b: &str,
    ) -> Option<DrugBankSeverity> {
        self.get_interaction(drug_a, drug_b).map(|i| i.severity)
    }

    /// Convenience alias used by composite scorer.
    pub fn lookup_severity(&self, drug_a: &str, drug_b: &str) -> Option<DrugBankSeverity> {
        self.get_interaction_severity(drug_a, drug_b)
    }

    /// All interactions involving a given drug.
    pub fn get_all_interactions_for(&self, drug: &str) -> Vec<&DrugBankInteraction> {
        let d = drug.to_lowercase();
        match self.drug_index.get(&d) {
            Some(keys) => keys
                .iter()
                .filter_map(|k| self.interactions.get(k))
                .collect(),
            None => Vec::new(),
        }
    }

    /// Whether any interaction record exists for the pair.
    pub fn has_interaction(&self, drug_a: &str, drug_b: &str) -> bool {
        self.get_interaction(drug_a, drug_b).is_some()
    }

    /// All *major* interactions in the database.
    pub fn get_major_interactions(&self) -> Vec<&DrugBankInteraction> {
        self.interactions
            .values()
            .filter(|i| i.severity.is_major())
            .collect()
    }

    /// Total number of interaction records.
    pub fn len(&self) -> usize {
        self.interactions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.interactions.is_empty()
    }

    /// Get the composite score for a drug pair (severity × evidence).
    pub fn composite_score(&self, drug_a: &str, drug_b: &str) -> f64 {
        self.get_interaction(drug_a, drug_b)
            .map(|i| i.composite_score())
            .unwrap_or(0.0)
    }

    // ── Default data ────────────────────────────────────────────────────

    fn load_defaults(&mut self) {
        // ── Warfarin interactions ──
        self.insert(interaction(
            "warfarin", "aspirin", DrugBankSeverity::Major, EvidenceLevel::Established,
            "Additive anticoagulant/antiplatelet effects increase bleeding risk",
            "Pharmacodynamic synergism on hemostasis pathways",
        ));
        self.insert(interaction(
            "warfarin", "fluconazole", DrugBankSeverity::Major, EvidenceLevel::Established,
            "Fluconazole inhibits CYP2C9 metabolism of S-warfarin, increasing INR and bleeding risk",
            "CYP2C9 inhibition",
        ));
        self.insert(interaction(
            "warfarin", "amiodarone", DrugBankSeverity::Major, EvidenceLevel::Established,
            "Amiodarone inhibits CYP2C9 and CYP3A4, increasing warfarin effect and bleeding risk",
            "CYP2C9/CYP3A4 inhibition",
        ));
        self.insert(interaction(
            "warfarin", "metronidazole", DrugBankSeverity::Major, EvidenceLevel::Established,
            "Metronidazole inhibits CYP2C9, markedly increasing warfarin anticoagulation",
            "CYP2C9 inhibition",
        ));
        self.insert(interaction(
            "warfarin", "ciprofloxacin", DrugBankSeverity::Major, EvidenceLevel::Probable,
            "Ciprofloxacin inhibits CYP1A2 and may displace warfarin from protein binding",
            "CYP1A2 inhibition and protein binding displacement",
        ));
        self.insert(interaction(
            "warfarin", "ibuprofen", DrugBankSeverity::Major, EvidenceLevel::Established,
            "NSAIDs increase bleeding risk through antiplatelet effects and GI mucosal damage",
            "Pharmacodynamic synergism and GI toxicity",
        ));
        self.insert(interaction(
            "warfarin", "trimethoprim_sulfamethoxazole", DrugBankSeverity::Major, EvidenceLevel::Established,
            "TMP-SMX inhibits CYP2C9 metabolism of S-warfarin, substantially raising INR",
            "CYP2C9 inhibition",
        ));
        self.insert(interaction(
            "warfarin", "phenytoin", DrugBankSeverity::Major, EvidenceLevel::Probable,
            "Complex bidirectional interaction: initial increase then decrease in warfarin effect",
            "CYP2C9 competition then CYP induction",
        ));
        self.insert(interaction(
            "warfarin", "rifampin", DrugBankSeverity::Major, EvidenceLevel::Established,
            "Rifampin powerfully induces CYP2C9/CYP3A4, dramatically reducing warfarin effect",
            "CYP2C9/CYP3A4 induction",
        ));
        self.insert(interaction(
            "warfarin", "carbamazepine", DrugBankSeverity::Major, EvidenceLevel::Established,
            "Carbamazepine induces CYP3A4 and CYP2C9, reducing warfarin anticoagulant effect",
            "CYP enzyme induction",
        ));
        self.insert(interaction(
            "warfarin", "erythromycin", DrugBankSeverity::Moderate, EvidenceLevel::Probable,
            "Erythromycin inhibits CYP3A4, modestly increasing warfarin levels",
            "CYP3A4 inhibition",
        ));
        self.insert(interaction(
            "warfarin", "omeprazole", DrugBankSeverity::Moderate, EvidenceLevel::Suspected,
            "Omeprazole may modestly inhibit CYP2C19 metabolism of R-warfarin",
            "CYP2C19 inhibition",
        ));
        self.insert(interaction(
            "warfarin", "clopidogrel", DrugBankSeverity::Major, EvidenceLevel::Established,
            "Dual anticoagulant/antiplatelet therapy significantly increases bleeding risk",
            "Pharmacodynamic synergism on hemostasis",
        ));

        // ── Statin interactions ──
        self.insert(interaction(
            "simvastatin", "amiodarone", DrugBankSeverity::Major, EvidenceLevel::Established,
            "Amiodarone inhibits CYP3A4, increasing simvastatin levels and rhabdomyolysis risk",
            "CYP3A4 inhibition",
        ));
        self.insert(interaction(
            "simvastatin", "clarithromycin", DrugBankSeverity::Major, EvidenceLevel::Established,
            "Clarithromycin strongly inhibits CYP3A4, greatly increasing simvastatin exposure",
            "CYP3A4 inhibition",
        ));
        self.insert(interaction(
            "simvastatin", "itraconazole", DrugBankSeverity::Major, EvidenceLevel::Established,
            "Itraconazole strongly inhibits CYP3A4, greatly increasing simvastatin exposure",
            "CYP3A4 inhibition",
        ));
        self.insert(interaction(
            "simvastatin", "diltiazem", DrugBankSeverity::Major, EvidenceLevel::Probable,
            "Diltiazem inhibits CYP3A4, increasing simvastatin levels; limit to 10 mg/day",
            "CYP3A4 inhibition",
        ));
        self.insert(interaction(
            "simvastatin", "amlodipine", DrugBankSeverity::Moderate, EvidenceLevel::Probable,
            "Amlodipine modestly inhibits CYP3A4; simvastatin dose should not exceed 20 mg",
            "CYP3A4 inhibition",
        ));
        self.insert(interaction(
            "simvastatin", "cyclosporine", DrugBankSeverity::Major, EvidenceLevel::Established,
            "Cyclosporine inhibits CYP3A4 and OATP1B1, dramatically increasing statin exposure",
            "CYP3A4 and OATP1B1 inhibition",
        ));
        self.insert(interaction(
            "simvastatin", "gemfibrozil", DrugBankSeverity::Major, EvidenceLevel::Established,
            "Gemfibrozil inhibits OATP1B1 and glucuronidation, increasing statin myopathy risk",
            "OATP1B1 inhibition and glucuronidation inhibition",
        ));
        self.insert(interaction(
            "atorvastatin", "clarithromycin", DrugBankSeverity::Major, EvidenceLevel::Established,
            "Clarithromycin inhibits CYP3A4, increasing atorvastatin exposure and myopathy risk",
            "CYP3A4 inhibition",
        ));
        self.insert(interaction(
            "atorvastatin", "cyclosporine", DrugBankSeverity::Major, EvidenceLevel::Established,
            "Cyclosporine increases statin exposure via CYP3A4 and transporter inhibition",
            "CYP3A4/OATP1B1 inhibition",
        ));
        self.insert(interaction(
            "rosuvastatin", "gemfibrozil", DrugBankSeverity::Major, EvidenceLevel::Probable,
            "Gemfibrozil increases rosuvastatin exposure via OATP1B1 inhibition",
            "OATP1B1 inhibition",
        ));
        self.insert(interaction(
            "lovastatin", "itraconazole", DrugBankSeverity::Major, EvidenceLevel::Established,
            "Itraconazole strongly inhibits CYP3A4, contraindicated with lovastatin",
            "CYP3A4 inhibition",
        ));

        // ── Methotrexate interactions ──
        self.insert(interaction(
            "methotrexate", "ibuprofen", DrugBankSeverity::Major, EvidenceLevel::Established,
            "NSAIDs reduce renal clearance of methotrexate, increasing toxicity risk",
            "Reduced renal tubular secretion of methotrexate",
        ));
        self.insert(interaction(
            "methotrexate", "naproxen", DrugBankSeverity::Major, EvidenceLevel::Established,
            "Naproxen reduces renal clearance of methotrexate, risk of pancytopenia",
            "Reduced renal tubular secretion",
        ));
        self.insert(interaction(
            "methotrexate", "trimethoprim_sulfamethoxazole", DrugBankSeverity::Major, EvidenceLevel::Established,
            "TMP-SMX inhibits folate pathway synergistically with methotrexate; fatal pancytopenia reported",
            "Synergistic folate antagonism and reduced renal clearance",
        ));
        self.insert(interaction(
            "methotrexate", "penicillin", DrugBankSeverity::Major, EvidenceLevel::Probable,
            "Penicillins reduce renal tubular secretion of methotrexate",
            "Competition for renal tubular secretion",
        ));
        self.insert(interaction(
            "methotrexate", "probenecid", DrugBankSeverity::Major, EvidenceLevel::Established,
            "Probenecid inhibits renal excretion of methotrexate, causing toxic accumulation",
            "Inhibition of renal tubular secretion",
        ));

        // ── Lithium interactions ──
        self.insert(interaction(
            "lithium", "lisinopril", DrugBankSeverity::Major, EvidenceLevel::Established,
            "ACE inhibitors reduce lithium clearance, risk of lithium toxicity",
            "Reduced renal lithium excretion via sodium depletion",
        ));
        self.insert(interaction(
            "lithium", "enalapril", DrugBankSeverity::Major, EvidenceLevel::Established,
            "ACE inhibitors reduce lithium clearance, risk of lithium toxicity",
            "Reduced renal lithium excretion",
        ));
        self.insert(interaction(
            "lithium", "losartan", DrugBankSeverity::Major, EvidenceLevel::Probable,
            "ARBs may reduce lithium clearance similar to ACE inhibitors",
            "Reduced renal lithium excretion",
        ));
        self.insert(interaction(
            "lithium", "hydrochlorothiazide", DrugBankSeverity::Major, EvidenceLevel::Established,
            "Thiazide diuretics reduce lithium clearance by 25%, risk of toxicity",
            "Enhanced proximal tubular reabsorption of lithium",
        ));
        self.insert(interaction(
            "lithium", "ibuprofen", DrugBankSeverity::Major, EvidenceLevel::Established,
            "NSAIDs reduce renal prostaglandin-mediated lithium excretion",
            "Reduced renal blood flow and lithium clearance",
        ));
        self.insert(interaction(
            "lithium", "naproxen", DrugBankSeverity::Major, EvidenceLevel::Established,
            "NSAIDs reduce lithium clearance by up to 20%",
            "Reduced renal prostaglandin-mediated excretion",
        ));

        // ── Digoxin interactions ──
        self.insert(interaction(
            "digoxin", "amiodarone", DrugBankSeverity::Major, EvidenceLevel::Established,
            "Amiodarone increases digoxin levels by 70-100% via P-gp inhibition; halve digoxin dose",
            "P-glycoprotein inhibition and reduced renal clearance",
        ));
        self.insert(interaction(
            "digoxin", "verapamil", DrugBankSeverity::Major, EvidenceLevel::Established,
            "Verapamil increases digoxin levels via P-gp inhibition; additive bradycardia",
            "P-glycoprotein inhibition and pharmacodynamic synergism",
        ));
        self.insert(interaction(
            "digoxin", "quinidine", DrugBankSeverity::Major, EvidenceLevel::Established,
            "Quinidine doubles digoxin levels via P-gp inhibition and reduced renal clearance",
            "P-glycoprotein inhibition",
        ));
        self.insert(interaction(
            "digoxin", "clarithromycin", DrugBankSeverity::Major, EvidenceLevel::Established,
            "Clarithromycin inhibits P-gp and kills gut flora that inactivate digoxin",
            "P-glycoprotein inhibition and altered gut flora",
        ));
        self.insert(interaction(
            "digoxin", "spironolactone", DrugBankSeverity::Moderate, EvidenceLevel::Probable,
            "Spironolactone may increase digoxin levels and interfere with digoxin assay",
            "Reduced renal clearance and assay interference",
        ));
        self.insert(interaction(
            "digoxin", "erythromycin", DrugBankSeverity::Moderate, EvidenceLevel::Probable,
            "Erythromycin inhibits P-gp and may increase digoxin levels",
            "P-glycoprotein inhibition",
        ));

        // ── Clopidogrel interactions ──
        self.insert(interaction(
            "clopidogrel", "omeprazole", DrugBankSeverity::Major, EvidenceLevel::Probable,
            "Omeprazole inhibits CYP2C19 activation of clopidogrel, reducing antiplatelet effect",
            "CYP2C19 inhibition of prodrug activation",
        ));
        self.insert(interaction(
            "clopidogrel", "esomeprazole", DrugBankSeverity::Major, EvidenceLevel::Probable,
            "Esomeprazole inhibits CYP2C19, reducing clopidogrel active metabolite formation",
            "CYP2C19 inhibition",
        ));
        self.insert(interaction(
            "clopidogrel", "fluconazole", DrugBankSeverity::Moderate, EvidenceLevel::Suspected,
            "Fluconazole inhibits CYP2C19, potentially reducing clopidogrel activation",
            "CYP2C19 inhibition",
        ));
        self.insert(interaction(
            "clopidogrel", "aspirin", DrugBankSeverity::Moderate, EvidenceLevel::Established,
            "Dual antiplatelet therapy increases bleeding risk but may be intentional (DAPT)",
            "Additive antiplatelet effect",
        ));

        // ── SSRI / serotonin interactions ──
        self.insert(interaction(
            "fluoxetine", "tramadol", DrugBankSeverity::Major, EvidenceLevel::Established,
            "Serotonin syndrome risk from combined serotonergic activity; fluoxetine also inhibits tramadol activation via CYP2D6",
            "Pharmacodynamic serotonin synergism and CYP2D6 inhibition",
        ));
        self.insert(interaction(
            "sertraline", "tramadol", DrugBankSeverity::Major, EvidenceLevel::Established,
            "Risk of serotonin syndrome; both drugs increase serotonergic activity",
            "Pharmacodynamic serotonin synergism",
        ));
        self.insert(interaction(
            "fluoxetine", "linezolid", DrugBankSeverity::Major, EvidenceLevel::Established,
            "Linezolid is a non-selective MAO inhibitor; contraindicated with SSRIs due to serotonin syndrome risk",
            "MAO inhibition combined with serotonin reuptake inhibition",
        ));
        self.insert(interaction(
            "sertraline", "linezolid", DrugBankSeverity::Major, EvidenceLevel::Established,
            "Serotonin syndrome risk; linezolid is a reversible MAO inhibitor",
            "MAO inhibition combined with SSRI",
        ));
        self.insert(interaction(
            "paroxetine", "tramadol", DrugBankSeverity::Major, EvidenceLevel::Established,
            "Serotonin syndrome risk; paroxetine also inhibits CYP2D6 activation of tramadol",
            "Serotonergic synergism and CYP2D6 inhibition",
        ));
        self.insert(interaction(
            "fluoxetine", "phenelzine", DrugBankSeverity::Major, EvidenceLevel::Established,
            "Potentially fatal serotonin syndrome; 5-week washout required",
            "Combined MAO inhibition and serotonin reuptake inhibition",
        ));
        self.insert(interaction(
            "fluoxetine", "metoprolol", DrugBankSeverity::Moderate, EvidenceLevel::Probable,
            "Fluoxetine inhibits CYP2D6, increasing metoprolol levels and risk of bradycardia",
            "CYP2D6 inhibition",
        ));
        self.insert(interaction(
            "paroxetine", "metoprolol", DrugBankSeverity::Moderate, EvidenceLevel::Probable,
            "Paroxetine inhibits CYP2D6, increasing metoprolol exposure",
            "CYP2D6 inhibition",
        ));
        self.insert(interaction(
            "fluoxetine", "tamoxifen", DrugBankSeverity::Major, EvidenceLevel::Established,
            "Fluoxetine inhibits CYP2D6, preventing tamoxifen conversion to active endoxifen",
            "CYP2D6 inhibition of prodrug activation",
        ));
        self.insert(interaction(
            "paroxetine", "tamoxifen", DrugBankSeverity::Major, EvidenceLevel::Established,
            "Paroxetine strongly inhibits CYP2D6, reducing tamoxifen efficacy in breast cancer",
            "CYP2D6 inhibition",
        ));

        // ── Potassium / hyperkalemia interactions ──
        self.insert(interaction(
            "spironolactone", "lisinopril", DrugBankSeverity::Major, EvidenceLevel::Established,
            "Dual RAAS blockade increases hyperkalemia risk, especially in CKD or diabetes",
            "Additive potassium retention",
        ));
        self.insert(interaction(
            "spironolactone", "losartan", DrugBankSeverity::Major, EvidenceLevel::Established,
            "Combined potassium-sparing diuretic and ARB increases hyperkalemia risk",
            "Additive potassium retention",
        ));
        self.insert(interaction(
            "spironolactone", "potassium_chloride", DrugBankSeverity::Major, EvidenceLevel::Established,
            "Potassium supplementation with potassium-sparing diuretic risks severe hyperkalemia",
            "Additive potassium load",
        ));
        self.insert(interaction(
            "lisinopril", "potassium_chloride", DrugBankSeverity::Major, EvidenceLevel::Probable,
            "ACE inhibitors reduce aldosterone, decreasing potassium excretion; supplementation risky",
            "Reduced aldosterone-mediated potassium excretion",
        ));
        self.insert(interaction(
            "trimethoprim_sulfamethoxazole", "spironolactone", DrugBankSeverity::Major, EvidenceLevel::Probable,
            "Both reduce potassium excretion; combined use associated with sudden cardiac death",
            "Additive potassium retention (ENaC blockade + aldosterone antagonism)",
        ));

        // ── Hypoglycemia interactions ──
        self.insert(interaction(
            "glipizide", "fluconazole", DrugBankSeverity::Major, EvidenceLevel::Established,
            "Fluconazole inhibits CYP2C9 metabolism of sulfonylureas, causing severe hypoglycemia",
            "CYP2C9 inhibition",
        ));
        self.insert(interaction(
            "glyburide", "fluconazole", DrugBankSeverity::Major, EvidenceLevel::Established,
            "Fluconazole inhibits CYP2C9, increasing glyburide levels and hypoglycemia risk",
            "CYP2C9 inhibition",
        ));
        self.insert(interaction(
            "insulin", "fluoxetine", DrugBankSeverity::Moderate, EvidenceLevel::Suspected,
            "SSRIs may potentiate hypoglycemic effect of insulin",
            "Possible enhanced insulin sensitivity",
        ));
        self.insert(interaction(
            "metformin", "contrast_dye", DrugBankSeverity::Major, EvidenceLevel::Established,
            "Iodinated contrast with metformin risks lactic acidosis in renal impairment",
            "Acute kidney injury reducing metformin clearance",
        ));

        // ── QT prolongation interactions ──
        self.insert(interaction(
            "amiodarone", "levofloxacin", DrugBankSeverity::Major, EvidenceLevel::Established,
            "Additive QTc prolongation; risk of torsades de pointes",
            "Dual hERG channel blockade",
        ));
        self.insert(interaction(
            "amiodarone", "sotalol", DrugBankSeverity::Major, EvidenceLevel::Established,
            "Severe QTc prolongation and risk of ventricular arrhythmia",
            "Dual potassium channel blockade",
        ));
        self.insert(interaction(
            "haloperidol", "methadone", DrugBankSeverity::Major, EvidenceLevel::Probable,
            "Both prolong QTc; additive risk of torsades de pointes",
            "Dual hERG channel blockade",
        ));
        self.insert(interaction(
            "erythromycin", "cisapride", DrugBankSeverity::Major, EvidenceLevel::Established,
            "Erythromycin inhibits CYP3A4 metabolism of cisapride and both prolong QT",
            "CYP3A4 inhibition and dual QT prolongation",
        ));
        self.insert(interaction(
            "ondansetron", "haloperidol", DrugBankSeverity::Moderate, EvidenceLevel::Probable,
            "Additive QTc prolongation at higher doses",
            "Dual hERG channel blockade",
        ));
        self.insert(interaction(
            "ciprofloxacin", "tizanidine", DrugBankSeverity::Major, EvidenceLevel::Established,
            "Ciprofloxacin inhibits CYP1A2, increasing tizanidine levels 10-fold; contraindicated",
            "CYP1A2 inhibition",
        ));

        // ── CNS depression interactions ──
        self.insert(interaction(
            "oxycodone", "benzodiazepine", DrugBankSeverity::Major, EvidenceLevel::Established,
            "Combined opioid and benzodiazepine use increases risk of fatal respiratory depression",
            "Additive CNS and respiratory depression",
        ));
        self.insert(interaction(
            "morphine", "diazepam", DrugBankSeverity::Major, EvidenceLevel::Established,
            "Opioid-benzodiazepine combination carries FDA black box warning for respiratory depression",
            "Additive CNS depression",
        ));
        self.insert(interaction(
            "gabapentin", "oxycodone", DrugBankSeverity::Major, EvidenceLevel::Probable,
            "Gabapentinoids potentiate opioid respiratory depression",
            "Additive CNS depression",
        ));
        self.insert(interaction(
            "pregabalin", "morphine", DrugBankSeverity::Major, EvidenceLevel::Probable,
            "Pregabalin increases opioid-related respiratory depression risk",
            "Additive CNS depression",
        ));
        self.insert(interaction(
            "zolpidem", "oxycodone", DrugBankSeverity::Major, EvidenceLevel::Established,
            "Combined sedative-hypnotic and opioid increases overdose risk",
            "Additive CNS depression",
        ));

        // ── Theophylline interactions ──
        self.insert(interaction(
            "theophylline", "ciprofloxacin", DrugBankSeverity::Major, EvidenceLevel::Established,
            "Ciprofloxacin inhibits CYP1A2, increasing theophylline levels and seizure risk",
            "CYP1A2 inhibition",
        ));
        self.insert(interaction(
            "theophylline", "erythromycin", DrugBankSeverity::Moderate, EvidenceLevel::Established,
            "Erythromycin inhibits CYP3A4, increasing theophylline levels",
            "CYP3A4 inhibition",
        ));
        self.insert(interaction(
            "theophylline", "fluvoxamine", DrugBankSeverity::Major, EvidenceLevel::Established,
            "Fluvoxamine strongly inhibits CYP1A2, dramatically increasing theophylline levels",
            "CYP1A2 inhibition",
        ));

        // ── Immunosuppressant interactions ──
        self.insert(interaction(
            "cyclosporine", "clarithromycin", DrugBankSeverity::Major, EvidenceLevel::Established,
            "Clarithromycin inhibits CYP3A4 and P-gp, increasing cyclosporine nephrotoxicity risk",
            "CYP3A4 and P-glycoprotein inhibition",
        ));
        self.insert(interaction(
            "tacrolimus", "fluconazole", DrugBankSeverity::Major, EvidenceLevel::Established,
            "Fluconazole inhibits CYP3A4, increasing tacrolimus levels and nephrotoxicity risk",
            "CYP3A4 inhibition",
        ));
        self.insert(interaction(
            "tacrolimus", "clarithromycin", DrugBankSeverity::Major, EvidenceLevel::Established,
            "Clarithromycin inhibits CYP3A4, increasing tacrolimus exposure",
            "CYP3A4 inhibition",
        ));
        self.insert(interaction(
            "cyclosporine", "rifampin", DrugBankSeverity::Major, EvidenceLevel::Established,
            "Rifampin induces CYP3A4, dramatically reducing cyclosporine levels and risking rejection",
            "CYP3A4 induction",
        ));

        // ── Anticoagulant + NSAID ──
        self.insert(interaction(
            "rivaroxaban", "aspirin", DrugBankSeverity::Major, EvidenceLevel::Established,
            "Combined DOAC and antiplatelet significantly increases major bleeding risk",
            "Pharmacodynamic anticoagulant/antiplatelet synergism",
        ));
        self.insert(interaction(
            "apixaban", "aspirin", DrugBankSeverity::Major, EvidenceLevel::Established,
            "Combined DOAC and antiplatelet increases major bleeding risk",
            "Pharmacodynamic synergism",
        ));
        self.insert(interaction(
            "rivaroxaban", "ibuprofen", DrugBankSeverity::Major, EvidenceLevel::Probable,
            "NSAID with DOAC significantly increases GI and intracranial bleeding risk",
            "Pharmacodynamic synergism and GI mucosal damage",
        ));
        self.insert(interaction(
            "dabigatran", "verapamil", DrugBankSeverity::Major, EvidenceLevel::Established,
            "Verapamil inhibits P-gp, increasing dabigatran levels; reduce dabigatran dose",
            "P-glycoprotein inhibition",
        ));
        self.insert(interaction(
            "dabigatran", "amiodarone", DrugBankSeverity::Major, EvidenceLevel::Probable,
            "Amiodarone inhibits P-gp, increasing dabigatran exposure",
            "P-glycoprotein inhibition",
        ));

        // ── Antiepileptic interactions ──
        self.insert(interaction(
            "phenytoin", "valproic_acid", DrugBankSeverity::Major, EvidenceLevel::Established,
            "Valproate displaces phenytoin from protein binding and inhibits its metabolism",
            "Protein binding displacement and CYP2C9 inhibition",
        ));
        self.insert(interaction(
            "carbamazepine", "erythromycin", DrugBankSeverity::Major, EvidenceLevel::Established,
            "Erythromycin inhibits CYP3A4, increasing carbamazepine levels and neurotoxicity risk",
            "CYP3A4 inhibition",
        ));
        self.insert(interaction(
            "carbamazepine", "valproic_acid", DrugBankSeverity::Major, EvidenceLevel::Probable,
            "Carbamazepine induces valproate metabolism; valproate inhibits carbamazepine-epoxide hydrolysis",
            "Bidirectional enzyme interaction",
        ));
        self.insert(interaction(
            "phenytoin", "isoniazid", DrugBankSeverity::Major, EvidenceLevel::Established,
            "Isoniazid inhibits CYP2C19, increasing phenytoin levels especially in slow acetylators",
            "CYP2C19 inhibition",
        ));

        // ── Antidepressant interactions ──
        self.insert(interaction(
            "venlafaxine", "tramadol", DrugBankSeverity::Major, EvidenceLevel::Probable,
            "Risk of serotonin syndrome; both increase serotonergic activity",
            "Serotonin reuptake inhibition synergism",
        ));
        self.insert(interaction(
            "duloxetine", "tramadol", DrugBankSeverity::Major, EvidenceLevel::Probable,
            "Serotonin syndrome risk from combined serotonergic activity",
            "Serotonin reuptake inhibition synergism",
        ));
        self.insert(interaction(
            "trazodone", "fluoxetine", DrugBankSeverity::Moderate, EvidenceLevel::Probable,
            "Fluoxetine inhibits CYP3A4/2D6, increasing trazodone levels; additive serotonin effect",
            "CYP inhibition and serotonergic synergism",
        ));

        // ── Antihypertensive interactions ──
        self.insert(interaction(
            "lisinopril", "losartan", DrugBankSeverity::Major, EvidenceLevel::Established,
            "Dual RAAS blockade increases hyperkalemia, hypotension, and renal failure risk",
            "Additive RAAS inhibition",
        ));
        self.insert(interaction(
            "amlodipine", "simvastatin", DrugBankSeverity::Moderate, EvidenceLevel::Probable,
            "Amlodipine moderately inhibits CYP3A4; simvastatin max 20 mg recommended",
            "CYP3A4 inhibition",
        ));

        // ── Metformin interactions ──
        self.insert(interaction(
            "metformin", "lisinopril", DrugBankSeverity::Minor, EvidenceLevel::Suspected,
            "ACE inhibitors may modestly enhance metformin glucose-lowering effect",
            "Possible enhanced insulin sensitivity",
        ));

        // ── Proton pump inhibitor interactions ──
        self.insert(interaction(
            "omeprazole", "clopidogrel", DrugBankSeverity::Major, EvidenceLevel::Probable,
            "Omeprazole inhibits CYP2C19 activation of clopidogrel",
            "CYP2C19 inhibition",
        ));
        self.insert(interaction(
            "omeprazole", "methotrexate", DrugBankSeverity::Moderate, EvidenceLevel::Suspected,
            "PPIs may reduce renal clearance of methotrexate, especially at high doses",
            "Inhibition of renal H+/K+ ATPase affecting MTX transport",
        ));

        // ── Additional clinically important interactions ──
        self.insert(interaction(
            "allopurinol", "azathioprine", DrugBankSeverity::Major, EvidenceLevel::Established,
            "Allopurinol inhibits xanthine oxidase metabolism of azathioprine, risking fatal pancytopenia",
            "Xanthine oxidase inhibition",
        ));
        self.insert(interaction(
            "allopurinol", "mercaptopurine", DrugBankSeverity::Major, EvidenceLevel::Established,
            "Allopurinol inhibits mercaptopurine metabolism; reduce dose by 75%",
            "Xanthine oxidase inhibition",
        ));
        self.insert(interaction(
            "sildenafil", "nitroglycerin", DrugBankSeverity::Major, EvidenceLevel::Established,
            "PDE5 inhibitors with nitrates cause severe potentially fatal hypotension; contraindicated",
            "Synergistic vasodilation via cGMP pathway",
        ));
        self.insert(interaction(
            "sildenafil", "isosorbide_dinitrate", DrugBankSeverity::Major, EvidenceLevel::Established,
            "PDE5 inhibitors contraindicated with any form of nitrate therapy",
            "Synergistic vasodilation via cGMP pathway",
        ));
        self.insert(interaction(
            "potassium_chloride", "amiloride", DrugBankSeverity::Major, EvidenceLevel::Established,
            "Potassium supplementation with potassium-sparing diuretics risks severe hyperkalemia",
            "Additive potassium retention",
        ));
        self.insert(interaction(
            "dofetilide", "verapamil", DrugBankSeverity::Major, EvidenceLevel::Established,
            "Verapamil increases dofetilide levels via OCT2 inhibition; contraindicated",
            "Renal cation transporter inhibition",
        ));
        self.insert(interaction(
            "clozapine", "ciprofloxacin", DrugBankSeverity::Major, EvidenceLevel::Established,
            "Ciprofloxacin inhibits CYP1A2, dramatically increasing clozapine levels",
            "CYP1A2 inhibition",
        ));
        self.insert(interaction(
            "clozapine", "fluvoxamine", DrugBankSeverity::Major, EvidenceLevel::Established,
            "Fluvoxamine strongly inhibits CYP1A2, increasing clozapine levels up to 10-fold",
            "CYP1A2 inhibition",
        ));
        self.insert(interaction(
            "colchicine", "clarithromycin", DrugBankSeverity::Major, EvidenceLevel::Established,
            "Clarithromycin inhibits CYP3A4 and P-gp; fatal colchicine toxicity reported",
            "CYP3A4 and P-glycoprotein inhibition",
        ));
        self.insert(interaction(
            "ergotamine", "clarithromycin", DrugBankSeverity::Major, EvidenceLevel::Established,
            "CYP3A4 inhibition increases ergotamine levels; risk of ergotism with gangrene",
            "CYP3A4 inhibition",
        ));
        self.insert(interaction(
            "pimozide", "clarithromycin", DrugBankSeverity::Major, EvidenceLevel::Established,
            "Clarithromycin increases pimozide levels via CYP3A4; risk of fatal QT prolongation",
            "CYP3A4 inhibition and additive QT prolongation",
        ));
        self.insert(interaction(
            "ranolazine", "ketoconazole", DrugBankSeverity::Major, EvidenceLevel::Established,
            "Ketoconazole strongly inhibits CYP3A4, increasing ranolazine levels; contraindicated",
            "CYP3A4 inhibition",
        ));
        self.insert(interaction(
            "fentanyl", "ritonavir", DrugBankSeverity::Major, EvidenceLevel::Established,
            "Ritonavir inhibits CYP3A4, increasing fentanyl levels and respiratory depression risk",
            "CYP3A4 inhibition",
        ));
        self.insert(interaction(
            "midazolam", "ketoconazole", DrugBankSeverity::Major, EvidenceLevel::Established,
            "Ketoconazole increases oral midazolam AUC 15-fold; contraindicated with oral form",
            "CYP3A4 inhibition",
        ));
        self.insert(interaction(
            "alprazolam", "ketoconazole", DrugBankSeverity::Major, EvidenceLevel::Established,
            "Ketoconazole inhibits CYP3A4, significantly increasing alprazolam exposure",
            "CYP3A4 inhibition",
        ));
        self.insert(interaction(
            "ticagrelor", "simvastatin", DrugBankSeverity::Moderate, EvidenceLevel::Probable,
            "Ticagrelor inhibits CYP3A4, increasing simvastatin levels; limit simvastatin to 40 mg",
            "CYP3A4 inhibition",
        ));
        self.insert(interaction(
            "dronedarone", "digoxin", DrugBankSeverity::Major, EvidenceLevel::Established,
            "Dronedarone inhibits P-gp, increasing digoxin levels; halve digoxin dose",
            "P-glycoprotein inhibition",
        ));
        self.insert(interaction(
            "ivabradine", "diltiazem", DrugBankSeverity::Major, EvidenceLevel::Established,
            "Diltiazem inhibits CYP3A4 and has additive heart rate lowering; contraindicated",
            "CYP3A4 inhibition and additive bradycardia",
        ));
        self.insert(interaction(
            "sacubitril_valsartan", "lisinopril", DrugBankSeverity::Major, EvidenceLevel::Established,
            "Sacubitril/valsartan with ACE inhibitor increases angioedema risk; 36-hour washout required",
            "Dual neprilysin and ACE inhibition",
        ));

        // ── Moderate / Minor interactions ──
        self.insert(interaction(
            "atorvastatin", "grapefruit", DrugBankSeverity::Moderate, EvidenceLevel::Established,
            "Grapefruit inhibits intestinal CYP3A4, modestly increasing atorvastatin levels",
            "Intestinal CYP3A4 inhibition",
        ));
        self.insert(interaction(
            "metformin", "alcohol", DrugBankSeverity::Moderate, EvidenceLevel::Probable,
            "Alcohol increases risk of metformin-associated lactic acidosis",
            "Impaired hepatic lactate clearance",
        ));
        self.insert(interaction(
            "levothyroxine", "calcium_carbonate", DrugBankSeverity::Moderate, EvidenceLevel::Established,
            "Calcium reduces levothyroxine absorption; separate by 4 hours",
            "Reduced GI absorption via chelation",
        ));
        self.insert(interaction(
            "levothyroxine", "iron", DrugBankSeverity::Moderate, EvidenceLevel::Established,
            "Iron reduces levothyroxine absorption; separate by 4 hours",
            "Reduced GI absorption via chelation",
        ));
        self.insert(interaction(
            "tetracycline", "calcium_carbonate", DrugBankSeverity::Moderate, EvidenceLevel::Established,
            "Divalent cations chelate tetracyclines, reducing absorption by up to 80%",
            "Chelation in GI tract",
        ));
        self.insert(interaction(
            "ciprofloxacin", "calcium_carbonate", DrugBankSeverity::Moderate, EvidenceLevel::Established,
            "Divalent cations reduce fluoroquinolone absorption; separate by 2 hours",
            "Chelation in GI tract",
        ));
        self.insert(interaction(
            "prednisone", "ibuprofen", DrugBankSeverity::Moderate, EvidenceLevel::Established,
            "Corticosteroids and NSAIDs have additive GI ulceration and bleeding risk",
            "Additive GI mucosal toxicity",
        ));
        self.insert(interaction(
            "metoprolol", "verapamil", DrugBankSeverity::Major, EvidenceLevel::Established,
            "Both depress cardiac conduction; risk of severe bradycardia, heart block, heart failure",
            "Additive negative chronotropic and inotropic effects",
        ));
        self.insert(interaction(
            "atenolol", "verapamil", DrugBankSeverity::Major, EvidenceLevel::Established,
            "Additive AV nodal depression; risk of heart block and asystole",
            "Additive negative chronotropic effects",
        ));
        self.insert(interaction(
            "propranolol", "verapamil", DrugBankSeverity::Major, EvidenceLevel::Established,
            "Severe bradycardia and heart failure risk from combined AV nodal and myocardial depression",
            "Additive negative chronotropic and inotropic effects",
        ));
    }
}

/// Produce a canonical (alphabetically sorted, lower-case) key.
fn canonical_key(drug_a: &str, drug_b: &str) -> (String, String) {
    let a = drug_a.to_lowercase();
    let b = drug_b.to_lowercase();
    if a <= b { (a, b) } else { (b, a) }
}

/// Helper to construct an interaction record.
fn interaction(
    drug_a: &str,
    drug_b: &str,
    severity: DrugBankSeverity,
    evidence: EvidenceLevel,
    description: &str,
    mechanism: &str,
) -> DrugBankInteraction {
    DrugBankInteraction::new(drug_a, drug_b, severity, description, mechanism, evidence)
}

// ──────────────────────────── Tests ──────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn db() -> DrugBankDatabase {
        DrugBankDatabase::with_defaults()
    }

    #[test]
    fn test_default_database_not_empty() {
        let d = db();
        assert!(d.len() >= 100, "Expected at least 100 interactions, got {}", d.len());
    }

    #[test]
    fn test_warfarin_aspirin_major() {
        let d = db();
        let sev = d.lookup_severity("warfarin", "aspirin");
        assert_eq!(sev, Some(DrugBankSeverity::Major));
    }

    #[test]
    fn test_case_insensitive() {
        let d = db();
        let sev = d.lookup_severity("Warfarin", "ASPIRIN");
        assert_eq!(sev, Some(DrugBankSeverity::Major));
    }

    #[test]
    fn test_symmetric_lookup() {
        let d = db();
        assert_eq!(
            d.lookup_severity("aspirin", "warfarin"),
            d.lookup_severity("warfarin", "aspirin"),
        );
    }

    #[test]
    fn test_unknown_pair_returns_none() {
        let d = db();
        assert!(d.lookup_severity("warfarin", "acetaminophen").is_none());
    }

    #[test]
    fn test_has_interaction() {
        let d = db();
        assert!(d.has_interaction("simvastatin", "amiodarone"));
        assert!(!d.has_interaction("water", "oxygen"));
    }

    #[test]
    fn test_get_all_interactions_for() {
        let d = db();
        let warfarin_interactions = d.get_all_interactions_for("warfarin");
        assert!(warfarin_interactions.len() >= 10, "Warfarin should have many interactions");
    }

    #[test]
    fn test_get_major_interactions() {
        let d = db();
        let majors = d.get_major_interactions();
        assert!(majors.len() >= 50, "Should have many major interactions");
        for m in &majors {
            assert_eq!(m.severity, DrugBankSeverity::Major);
        }
    }

    #[test]
    fn test_composite_score() {
        let d = db();
        let score = d.composite_score("warfarin", "aspirin");
        // Major (1.0) × Established (1.0) = 1.0
        assert!((score - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_composite_score_unknown_pair() {
        let d = db();
        assert!((d.composite_score("water", "nothing") - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_severity_score_ordering() {
        assert!(DrugBankSeverity::Minor.score() < DrugBankSeverity::Moderate.score());
        assert!(DrugBankSeverity::Moderate.score() < DrugBankSeverity::Major.score());
    }

    #[test]
    fn test_evidence_weight_ordering() {
        assert!(EvidenceLevel::Unlikely.weight() < EvidenceLevel::Possible.weight());
        assert!(EvidenceLevel::Possible.weight() < EvidenceLevel::Suspected.weight());
        assert!(EvidenceLevel::Suspected.weight() < EvidenceLevel::Probable.weight());
        assert!(EvidenceLevel::Probable.weight() < EvidenceLevel::Established.weight());
    }

    #[test]
    fn test_interaction_involves() {
        let i = DrugBankInteraction::new(
            "warfarin", "aspirin", DrugBankSeverity::Major, "test", "test", EvidenceLevel::Established,
        );
        assert!(i.involves("warfarin"));
        assert!(i.involves("Aspirin"));
        assert!(!i.involves("metformin"));
    }

    #[test]
    fn test_interaction_pair_key_canonical() {
        let i = DrugBankInteraction::new(
            "zolpidem", "aspirin", DrugBankSeverity::Minor, "", "", EvidenceLevel::Possible,
        );
        let (a, b) = i.pair_key();
        assert!(a <= b, "Pair key should be alphabetically ordered");
    }

    #[test]
    fn test_insert_custom_interaction() {
        let mut d = DrugBankDatabase::empty();
        d.insert(DrugBankInteraction::new(
            "drugA", "drugB", DrugBankSeverity::Moderate, "desc", "mech", EvidenceLevel::Suspected,
        ));
        assert_eq!(d.len(), 1);
        assert!(d.has_interaction("drugA", "drugB"));
        assert!(d.has_interaction("drugB", "drugA"));
    }

    #[test]
    fn test_drugbank_entry() {
        let mut entry = DrugBankEntry::new("DB00001", "Warfarin");
        entry.add_interaction(DrugBankInteraction::new(
            "warfarin", "aspirin", DrugBankSeverity::Major, "test", "test", EvidenceLevel::Established,
        ));
        assert_eq!(entry.interaction_count(), 1);
        assert_eq!(entry.major_interactions().len(), 1);
    }

    #[test]
    fn test_severity_display() {
        assert_eq!(format!("{}", DrugBankSeverity::Major), "Major");
        assert_eq!(format!("{}", DrugBankSeverity::Unknown), "Unknown");
    }

    #[test]
    fn test_evidence_display() {
        assert_eq!(format!("{}", EvidenceLevel::Established), "Established");
        assert_eq!(format!("{}", EvidenceLevel::Unlikely), "Unlikely");
    }

    #[test]
    fn test_simvastatin_clarithromycin() {
        let d = db();
        let i = d.get_interaction("simvastatin", "clarithromycin").unwrap();
        assert_eq!(i.severity, DrugBankSeverity::Major);
        assert!(i.mechanism.contains("CYP3A4"));
    }

    #[test]
    fn test_clopidogrel_omeprazole() {
        let d = db();
        let sev = d.lookup_severity("clopidogrel", "omeprazole");
        assert_eq!(sev, Some(DrugBankSeverity::Major));
    }

    #[test]
    fn test_lithium_hydrochlorothiazide() {
        let d = db();
        let i = d.get_interaction("lithium", "hydrochlorothiazide").unwrap();
        assert_eq!(i.severity, DrugBankSeverity::Major);
        assert_eq!(i.evidence_level, EvidenceLevel::Established);
    }

    #[test]
    fn test_fluoxetine_tramadol_serotonin() {
        let d = db();
        let i = d.get_interaction("fluoxetine", "tramadol").unwrap();
        assert_eq!(i.severity, DrugBankSeverity::Major);
        assert!(i.description.to_lowercase().contains("serotonin"));
    }

    #[test]
    fn test_severity_is_at_least_moderate() {
        assert!(DrugBankSeverity::Major.is_at_least_moderate());
        assert!(DrugBankSeverity::Moderate.is_at_least_moderate());
        assert!(!DrugBankSeverity::Minor.is_at_least_moderate());
        assert!(!DrugBankSeverity::Unknown.is_at_least_moderate());
    }
}
