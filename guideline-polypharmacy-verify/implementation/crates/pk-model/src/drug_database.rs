//! Drug database with realistic pharmacokinetic parameters.
//!
//! Contains ~30 commonly co-prescribed drugs with published PK values,
//! CYP metabolism routes, interaction data, and therapeutic windows.

use std::fmt;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use guardpharma_types::{DrugId, CypEnzyme, InhibitionType, Severity};
use guardpharma_types::drug::{DrugClass, DrugRoute, TherapeuticWindow, ToxicThreshold};

// ---------------------------------------------------------------------------
// DrugPkEntry
// ---------------------------------------------------------------------------

/// Complete PK profile for a single drug.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrugPkEntry {
    pub id: DrugId,
    pub generic_name: String,
    pub drug_class: DrugClass,
    pub route: DrugRoute,
    pub molecular_weight: f64,
    pub clearance: f64,           // L/h
    pub volume_distribution: f64, // L
    pub bioavailability: f64,     // fraction 0-1
    pub half_life: f64,           // hours
    pub tmax: f64,                // hours to peak
    pub protein_binding: f64,     // fraction 0-1
    pub therapeutic_window: TherapeuticWindow,
    pub toxic_threshold: Option<ToxicThreshold>,
    pub cyp_metabolism: Vec<CypMetabolismEntry>,
    pub cyp_inhibition: Vec<CypInhibitionEntry>,
    pub cyp_induction: Vec<CypInductionEntry>,
    pub renal_elimination_fraction: f64,
    pub hepatic_extraction_ratio: f64,
    pub typical_dose_mg: f64,
    pub typical_interval_h: f64,
    pub clearance_cv: f64,        // coefficient of variation
    pub volume_cv: f64,
}

impl DrugPkEntry {
    pub fn elimination_rate_constant(&self) -> f64 {
        0.693 / self.half_life
    }

    pub fn predicted_css_avg(&self) -> f64 {
        self.bioavailability * self.typical_dose_mg
            / (self.clearance * self.typical_interval_h)
    }

    pub fn is_within_therapeutic_window(&self, conc: f64) -> bool {
        self.therapeutic_window.contains(conc)
    }

    pub fn primary_cyp(&self) -> Option<CypEnzyme> {
        self.cyp_metabolism.iter()
            .max_by(|a, b| a.fraction_metabolized.partial_cmp(&b.fraction_metabolized).unwrap())
            .map(|e| e.enzyme)
    }
}

impl fmt::Display for DrugPkEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} (CL={:.1} L/h, V={:.1} L, t½={:.1} h, F={:.0}%)",
               self.generic_name, self.clearance, self.volume_distribution,
               self.half_life, self.bioavailability * 100.0)
    }
}

// ---------------------------------------------------------------------------
// Supporting types
// ---------------------------------------------------------------------------

/// CYP metabolism route entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CypMetabolismEntry {
    pub enzyme: CypEnzyme,
    pub fraction_metabolized: f64,
}

/// CYP inhibition effect from a drug.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CypInhibitionEntry {
    pub enzyme: CypEnzyme,
    pub inhibition_type: InhibitionType,
    pub ki: f64,
}

/// CYP induction effect from a drug.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CypInductionEntry {
    pub enzyme: CypEnzyme,
    pub emax: f64,
    pub ec50: f64,
}

// ---------------------------------------------------------------------------
// DrugInteractionEntry
// ---------------------------------------------------------------------------

/// Known drug-drug interaction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrugInteractionEntry {
    pub perpetrator: DrugId,
    pub victim: DrugId,
    pub mechanism: String,
    pub severity: Severity,
    pub auc_fold_change: f64,
    pub clinical_note: String,
}

// ---------------------------------------------------------------------------
// DrugDatabase
// ---------------------------------------------------------------------------

/// In-memory drug database.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrugDatabase {
    pub drugs: IndexMap<DrugId, DrugPkEntry>,
    pub interactions: Vec<DrugInteractionEntry>,
}

impl DrugDatabase {
    pub fn new() -> Self {
        Self { drugs: IndexMap::new(), interactions: Vec::new() }
    }

    pub fn add_drug(&mut self, entry: DrugPkEntry) {
        self.drugs.insert(entry.id.clone(), entry);
    }

    pub fn add_interaction(&mut self, entry: DrugInteractionEntry) {
        self.interactions.push(entry);
    }

    pub fn get_drug(&self, id: &DrugId) -> Option<&DrugPkEntry> {
        self.drugs.get(id)
    }

    pub fn get_drug_by_name(&self, name: &str) -> Option<&DrugPkEntry> {
        let id = DrugId::from_name(name);
        self.drugs.get(&id)
    }

    pub fn list_drugs(&self) -> Vec<&DrugPkEntry> {
        self.drugs.values().collect()
    }

    pub fn drug_count(&self) -> usize { self.drugs.len() }

    /// Drugs metabolised by a given CYP enzyme.
    pub fn drugs_metabolized_by(&self, enzyme: CypEnzyme) -> Vec<&DrugPkEntry> {
        self.drugs.values()
            .filter(|d| d.cyp_metabolism.iter().any(|m| m.enzyme == enzyme))
            .collect()
    }

    /// Drugs that inhibit a given CYP enzyme.
    pub fn drugs_inhibiting(&self, enzyme: CypEnzyme) -> Vec<&DrugPkEntry> {
        self.drugs.values()
            .filter(|d| d.cyp_inhibition.iter().any(|i| i.enzyme == enzyme))
            .collect()
    }

    /// Drugs that induce a given CYP enzyme.
    pub fn drugs_inducing(&self, enzyme: CypEnzyme) -> Vec<&DrugPkEntry> {
        self.drugs.values()
            .filter(|d| d.cyp_induction.iter().any(|i| i.enzyme == enzyme))
            .collect()
    }

    /// Drugs sharing any CYP enzyme for metabolism.
    pub fn drugs_sharing_enzyme(&self, drug_id: &DrugId) -> Vec<&DrugPkEntry> {
        let drug = match self.drugs.get(drug_id) {
            Some(d) => d,
            None => return Vec::new(),
        };
        let enzymes: Vec<CypEnzyme> = drug.cyp_metabolism.iter().map(|m| m.enzyme).collect();
        self.drugs.values()
            .filter(|d| {
                d.id != *drug_id
                    && d.cyp_metabolism.iter().any(|m| enzymes.contains(&m.enzyme))
            })
            .collect()
    }

    /// Known interaction pairs for a set of drugs.
    pub fn get_interaction_pairs(&self, drug_ids: &[DrugId]) -> Vec<&DrugInteractionEntry> {
        let set: std::collections::HashSet<&DrugId> = drug_ids.iter().collect();
        self.interactions.iter()
            .filter(|i| set.contains(&i.perpetrator) && set.contains(&i.victim))
            .collect()
    }

    /// Drugs within a given class.
    pub fn drugs_by_class(&self, class: DrugClass) -> Vec<&DrugPkEntry> {
        self.drugs.values().filter(|d| d.drug_class == class).collect()
    }
}

impl Default for DrugDatabase {
    fn default() -> Self { Self::new() }
}

// ---------------------------------------------------------------------------
// Drug builder helper
// ---------------------------------------------------------------------------

fn drug(
    name: &str, class: DrugClass, mw: f64,
    cl: f64, v: f64, f_bio: f64, t_half: f64, tmax: f64, pb: f64,
    tw_min: f64, tw_max: f64,
    dose: f64, interval: f64,
    renal_frac: f64, hepatic_er: f64,
    cl_cv: f64, v_cv: f64,
) -> DrugPkEntry {
    DrugPkEntry {
        id: DrugId::from_name(name),
        generic_name: name.to_string(),
        drug_class: class,
        route: DrugRoute::Oral,
        molecular_weight: mw,
        clearance: cl,
        volume_distribution: v,
        bioavailability: f_bio,
        half_life: t_half,
        tmax,
        protein_binding: pb,
        therapeutic_window: TherapeuticWindow::new(tw_min, tw_max, "mcg/mL"),
        toxic_threshold: None,
        cyp_metabolism: Vec::new(),
        cyp_inhibition: Vec::new(),
        cyp_induction: Vec::new(),
        renal_elimination_fraction: renal_frac,
        hepatic_extraction_ratio: hepatic_er,
        typical_dose_mg: dose,
        typical_interval_h: interval,
        clearance_cv: cl_cv,
        volume_cv: v_cv,
    }
}

fn met(enzyme: CypEnzyme, fm: f64) -> CypMetabolismEntry {
    CypMetabolismEntry { enzyme, fraction_metabolized: fm }
}

fn inh(enzyme: CypEnzyme, itype: InhibitionType, ki: f64) -> CypInhibitionEntry {
    CypInhibitionEntry { enzyme, inhibition_type: itype, ki }
}

fn ind(enzyme: CypEnzyme, emax: f64, ec50: f64) -> CypInductionEntry {
    CypInductionEntry { enzyme, emax, ec50 }
}

// ---------------------------------------------------------------------------
// Built-in database
// ---------------------------------------------------------------------------

/// Build a drug database with ~30 drugs and realistic PK parameters.
pub fn build_default_database() -> DrugDatabase {
    let mut db = DrugDatabase::new();
    let c = InhibitionType::Competitive;
    let nc = InhibitionType::NonCompetitive;

    // 1. Warfarin (anticoagulant)
    let mut d = drug("warfarin", DrugClass::Anticoagulant, 308.3,
        0.2, 10.0, 0.99, 40.0, 4.0, 0.99,
        1.0, 4.0, 5.0, 24.0, 0.01, 0.0, 0.3, 0.2);
    d.cyp_metabolism = vec![met(CypEnzyme::CYP2C9, 0.85), met(CypEnzyme::CYP3A4, 0.10)];
    db.add_drug(d);

    // 2. Simvastatin (statin)
    let mut d = drug("simvastatin", DrugClass::Statin, 418.6,
        30.0, 150.0, 0.05, 3.0, 1.5, 0.95,
        0.005, 0.02, 40.0, 24.0, 0.13, 0.8, 0.4, 0.3);
    d.cyp_metabolism = vec![met(CypEnzyme::CYP3A4, 0.90)];
    db.add_drug(d);

    // 3. Atorvastatin (statin)
    let mut d = drug("atorvastatin", DrugClass::Statin, 558.6,
        38.0, 381.0, 0.14, 14.0, 2.0, 0.98,
        0.01, 0.08, 20.0, 24.0, 0.02, 0.6, 0.3, 0.3);
    d.cyp_metabolism = vec![met(CypEnzyme::CYP3A4, 0.70), met(CypEnzyme::CYP2C8, 0.20)];
    db.add_drug(d);

    // 4. Metformin (antidiabetic)
    let mut d = drug("metformin", DrugClass::Antidiabetic, 129.2,
        26.5, 654.0, 0.55, 5.0, 2.5, 0.01,
        0.5, 5.0, 500.0, 12.0, 0.90, 0.0, 0.3, 0.2);
    d.cyp_metabolism = vec![];
    db.add_drug(d);

    // 5. Amlodipine (calcium channel blocker)
    let mut d = drug("amlodipine", DrugClass::Antihypertensive, 408.9,
        7.0, 1050.0, 0.64, 40.0, 8.0, 0.93,
        0.003, 0.01, 5.0, 24.0, 0.10, 0.2, 0.3, 0.3);
    d.cyp_metabolism = vec![met(CypEnzyme::CYP3A4, 0.90)];
    db.add_drug(d);

    // 6. Metoprolol (beta blocker)
    let mut d = drug("metoprolol", DrugClass::BetaBlocker, 267.4,
        63.0, 290.0, 0.50, 3.5, 1.5, 0.12,
        0.02, 0.20, 50.0, 12.0, 0.05, 0.7, 0.4, 0.3);
    d.cyp_metabolism = vec![met(CypEnzyme::CYP2D6, 0.80)];
    db.add_drug(d);

    // 7. Omeprazole (proton pump inhibitor)
    let mut d = drug("omeprazole", DrugClass::PPI, 345.4,
        30.0, 20.0, 0.40, 1.0, 1.5, 0.95,
        0.1, 2.0, 20.0, 24.0, 0.0, 0.9, 0.4, 0.3);
    d.cyp_metabolism = vec![met(CypEnzyme::CYP2C19, 0.80), met(CypEnzyme::CYP3A4, 0.15)];
    d.cyp_inhibition = vec![inh(CypEnzyme::CYP2C19, c, 2.0)];
    db.add_drug(d);

    // 8. Fluoxetine (SSRI)
    let mut d = drug("fluoxetine", DrugClass::Antidepressant, 309.3,
        40.0, 2500.0, 0.72, 48.0, 6.0, 0.95,
        0.12, 0.50, 20.0, 24.0, 0.11, 0.2, 0.35, 0.3);
    d.cyp_metabolism = vec![met(CypEnzyme::CYP2D6, 0.50), met(CypEnzyme::CYP2C9, 0.30)];
    d.cyp_inhibition = vec![inh(CypEnzyme::CYP2D6, c, 0.5), inh(CypEnzyme::CYP2C19, c, 5.0)];
    db.add_drug(d);

    // 9. Paroxetine (SSRI)
    let mut d = drug("paroxetine", DrugClass::Antidepressant, 329.4,
        50.0, 1700.0, 0.50, 20.0, 5.0, 0.95,
        0.02, 0.06, 20.0, 24.0, 0.02, 0.5, 0.4, 0.3);
    d.cyp_metabolism = vec![met(CypEnzyme::CYP2D6, 0.70)];
    d.cyp_inhibition = vec![inh(CypEnzyme::CYP2D6, c, 0.15)];
    db.add_drug(d);

    // 10. Codeine (opioid)
    let mut d = drug("codeine", DrugClass::Analgesic, 299.4,
        54.0, 180.0, 0.53, 3.0, 1.0, 0.07,
        0.03, 0.20, 30.0, 6.0, 0.10, 0.5, 0.3, 0.25);
    d.cyp_metabolism = vec![met(CypEnzyme::CYP2D6, 0.10), met(CypEnzyme::CYP3A4, 0.70)];
    db.add_drug(d);

    // 11. Ketoconazole (antifungal)
    let mut d = drug("ketoconazole", DrugClass::Antifungal, 531.4,
        8.4, 140.0, 0.76, 8.0, 2.0, 0.99,
        1.0, 10.0, 200.0, 12.0, 0.13, 0.5, 0.3, 0.3);
    d.cyp_metabolism = vec![met(CypEnzyme::CYP3A4, 0.95)];
    d.cyp_inhibition = vec![inh(CypEnzyme::CYP3A4, c, 0.015)];
    db.add_drug(d);

    // 12. Fluconazole (antifungal)
    let mut d = drug("fluconazole", DrugClass::Antifungal, 306.3,
        1.5, 50.0, 0.90, 30.0, 2.0, 0.12,
        3.0, 30.0, 200.0, 24.0, 0.80, 0.04, 0.2, 0.2);
    d.cyp_inhibition = vec![
        inh(CypEnzyme::CYP2C9, c, 7.0),
        inh(CypEnzyme::CYP2C19, c, 3.0),
        inh(CypEnzyme::CYP3A4, c, 10.0),
    ];
    db.add_drug(d);

    // 13. Rifampin (antibiotic / inducer)
    let mut d = drug("rifampin", DrugClass::Antibiotic, 822.9,
        7.0, 68.0, 0.68, 3.5, 2.0, 0.80,
        4.0, 24.0, 600.0, 24.0, 0.15, 0.7, 0.3, 0.3);
    d.cyp_metabolism = vec![met(CypEnzyme::CYP3A4, 0.40)];
    d.cyp_induction = vec![
        ind(CypEnzyme::CYP3A4, 10.0, 0.5),
        ind(CypEnzyme::CYP2C9, 3.0, 1.0),
        ind(CypEnzyme::CYP2C19, 4.0, 0.8),
    ];
    db.add_drug(d);

    // 14. Cyclosporine (immunosuppressant)
    let mut d = drug("cyclosporine", DrugClass::Immunosuppressant, 1202.6,
        24.0, 350.0, 0.30, 6.0, 2.0, 0.90,
        0.1, 0.4, 100.0, 12.0, 0.06, 0.6, 0.4, 0.3);
    d.cyp_metabolism = vec![met(CypEnzyme::CYP3A4, 0.95)];
    d.cyp_inhibition = vec![inh(CypEnzyme::CYP3A4, nc, 1.0)];
    db.add_drug(d);

    // 15. Tacrolimus (immunosuppressant)
    let mut d = drug("tacrolimus", DrugClass::Immunosuppressant, 804.0,
        2.2, 75.0, 0.25, 12.0, 2.0, 0.99,
        0.005, 0.02, 2.0, 12.0, 0.01, 0.7, 0.4, 0.35);
    d.cyp_metabolism = vec![met(CypEnzyme::CYP3A4, 0.95)];
    db.add_drug(d);

    // 16. Ciprofloxacin (antibiotic)
    let mut d = drug("ciprofloxacin", DrugClass::Antibiotic, 331.3,
        36.0, 175.0, 0.70, 4.0, 1.5, 0.30,
        1.0, 5.0, 500.0, 12.0, 0.45, 0.15, 0.3, 0.25);
    d.cyp_inhibition = vec![inh(CypEnzyme::CYP1A2, c, 4.0)];
    db.add_drug(d);

    // 17. Theophylline (bronchodilator)
    let mut d = drug("theophylline", DrugClass::Bronchodilator, 180.2,
        3.6, 35.0, 0.96, 8.0, 2.0, 0.40,
        5.0, 15.0, 300.0, 12.0, 0.10, 0.3, 0.3, 0.2);
    d.cyp_metabolism = vec![met(CypEnzyme::CYP1A2, 0.70), met(CypEnzyme::CYP2E1, 0.15)];
    db.add_drug(d);

    // 18. Clopidogrel (antiplatelet)
    let mut d = drug("clopidogrel", DrugClass::Antiplatelet, 321.8,
        50.0, 125.0, 0.50, 6.0, 1.0, 0.98,
        0.001, 0.01, 75.0, 24.0, 0.50, 0.4, 0.35, 0.3);
    d.cyp_metabolism = vec![met(CypEnzyme::CYP2C19, 0.50), met(CypEnzyme::CYP3A4, 0.30)];
    db.add_drug(d);

    // 19. Diazepam (benzodiazepine)
    let mut d = drug("diazepam", DrugClass::Sedative, 284.7,
        1.6, 70.0, 0.99, 40.0, 1.0, 0.98,
        0.2, 1.5, 5.0, 8.0, 0.0, 0.3, 0.3, 0.25);
    d.cyp_metabolism = vec![met(CypEnzyme::CYP3A4, 0.55), met(CypEnzyme::CYP2C19, 0.40)];
    db.add_drug(d);

    // 20. Midazolam (benzodiazepine)
    let mut d = drug("midazolam", DrugClass::Sedative, 325.8,
        25.0, 54.0, 0.36, 2.5, 0.5, 0.97,
        0.04, 0.25, 7.5, 8.0, 0.01, 0.5, 0.35, 0.3);
    d.cyp_metabolism = vec![met(CypEnzyme::CYP3A4, 0.95)];
    db.add_drug(d);

    // 21. Phenytoin (antiepileptic)
    let mut d = drug("phenytoin", DrugClass::Antiepileptic, 252.3,
        4.0, 45.0, 0.90, 22.0, 4.0, 0.90,
        10.0, 20.0, 300.0, 24.0, 0.05, 0.4, 0.4, 0.3);
    d.cyp_metabolism = vec![met(CypEnzyme::CYP2C9, 0.90), met(CypEnzyme::CYP2C19, 0.05)];
    d.cyp_induction = vec![ind(CypEnzyme::CYP3A4, 5.0, 5.0), ind(CypEnzyme::CYP2C9, 2.0, 6.0)];
    db.add_drug(d);

    // 22. Carbamazepine (antiepileptic)
    let mut d = drug("carbamazepine", DrugClass::Antiepileptic, 236.3,
        5.0, 98.0, 0.75, 16.0, 6.0, 0.76,
        4.0, 12.0, 200.0, 12.0, 0.03, 0.5, 0.35, 0.3);
    d.cyp_metabolism = vec![met(CypEnzyme::CYP3A4, 0.75)];
    d.cyp_induction = vec![ind(CypEnzyme::CYP3A4, 8.0, 3.0), ind(CypEnzyme::CYP2C9, 2.5, 4.0)];
    db.add_drug(d);

    // 23. Amiodarone (antiarrhythmic)
    let mut d = drug("amiodarone", DrugClass::Antiarrhythmic, 645.3,
        6.3, 5000.0, 0.50, 800.0, 5.0, 0.96,
        0.5, 2.5, 200.0, 24.0, 0.0, 0.1, 0.35, 0.3);
    d.cyp_metabolism = vec![met(CypEnzyme::CYP3A4, 0.50), met(CypEnzyme::CYP2C8, 0.30)];
    d.cyp_inhibition = vec![
        inh(CypEnzyme::CYP3A4, nc, 1.5),
        inh(CypEnzyme::CYP2D6, c, 3.0),
        inh(CypEnzyme::CYP2C9, c, 8.0),
    ];
    db.add_drug(d);

    // 24. Digoxin (cardiac glycoside)
    let mut d = drug("digoxin", DrugClass::CardiacGlycoside, 780.9,
        7.0, 475.0, 0.70, 39.0, 2.0, 0.25,
        0.0008, 0.002, 0.25, 24.0, 0.70, 0.0, 0.25, 0.25);
    d.cyp_metabolism = vec![];
    db.add_drug(d);

    // 25. Lisinopril (ACE inhibitor)
    let mut d = drug("lisinopril", DrugClass::ACEInhibitor, 405.5,
        4.5, 84.0, 0.25, 12.0, 6.0, 0.0,
        0.01, 0.10, 10.0, 24.0, 1.0, 0.0, 0.3, 0.25);
    d.cyp_metabolism = vec![];
    db.add_drug(d);

    // 26. Ibuprofen (NSAID)
    let mut d = drug("ibuprofen", DrugClass::NSAID, 206.3,
        3.5, 10.0, 0.80, 2.0, 1.5, 0.99,
        10.0, 40.0, 400.0, 8.0, 0.01, 0.3, 0.25, 0.2);
    d.cyp_metabolism = vec![met(CypEnzyme::CYP2C9, 0.60), met(CypEnzyme::CYP2C19, 0.20)];
    db.add_drug(d);

    // 27. Acetaminophen (analgesic)
    let mut d = drug("acetaminophen", DrugClass::Analgesic, 151.2,
        20.0, 67.0, 0.85, 2.5, 1.0, 0.20,
        5.0, 20.0, 1000.0, 6.0, 0.05, 0.5, 0.25, 0.2);
    d.toxic_threshold = Some(ToxicThreshold::new(150.0, "mcg/mL", "hepatotoxicity"));
    d.cyp_metabolism = vec![met(CypEnzyme::CYP2E1, 0.10), met(CypEnzyme::CYP1A2, 0.05)];
    db.add_drug(d);

    // 28. Methotrexate (antimetabolite)
    let mut d = drug("methotrexate", DrugClass::Antimetabolite, 454.4,
        8.0, 18.0, 0.64, 6.0, 2.0, 0.50,
        0.01, 0.10, 15.0, 168.0, 0.90, 0.05, 0.35, 0.3);
    d.cyp_metabolism = vec![];
    db.add_drug(d);

    // 29. Erythromycin (macrolide antibiotic)
    let mut d = drug("erythromycin", DrugClass::Antibiotic, 733.9,
        36.0, 70.0, 0.35, 1.5, 2.0, 0.80,
        0.5, 4.0, 500.0, 6.0, 0.05, 0.5, 0.3, 0.25);
    d.cyp_metabolism = vec![met(CypEnzyme::CYP3A4, 0.80)];
    d.cyp_inhibition = vec![inh(CypEnzyme::CYP3A4, InhibitionType::MechanismBased, 3.5)];
    db.add_drug(d);

    // 30. Verapamil (calcium channel blocker)
    let mut d = drug("verapamil", DrugClass::Antihypertensive, 454.6,
        65.0, 300.0, 0.22, 6.0, 1.5, 0.90,
        0.05, 0.30, 80.0, 8.0, 0.05, 0.8, 0.35, 0.3);
    d.cyp_metabolism = vec![met(CypEnzyme::CYP3A4, 0.80), met(CypEnzyme::CYP1A2, 0.10)];
    d.cyp_inhibition = vec![inh(CypEnzyme::CYP3A4, c, 2.0)];
    db.add_drug(d);

    // Add known interactions
    db.add_interaction(DrugInteractionEntry {
        perpetrator: DrugId::from_name("fluconazole"),
        victim: DrugId::from_name("warfarin"),
        mechanism: "CYP2C9 inhibition".into(),
        severity: Severity::Major,
        auc_fold_change: 2.0,
        clinical_note: "Monitor INR closely".into(),
    });
    db.add_interaction(DrugInteractionEntry {
        perpetrator: DrugId::from_name("ketoconazole"),
        victim: DrugId::from_name("simvastatin"),
        mechanism: "CYP3A4 inhibition".into(),
        severity: Severity::Contraindicated,
        auc_fold_change: 12.0,
        clinical_note: "Contraindicated: rhabdomyolysis risk".into(),
    });
    db.add_interaction(DrugInteractionEntry {
        perpetrator: DrugId::from_name("rifampin"),
        victim: DrugId::from_name("cyclosporine"),
        mechanism: "CYP3A4 induction".into(),
        severity: Severity::Major,
        auc_fold_change: 0.1,
        clinical_note: "Transplant rejection risk".into(),
    });
    db.add_interaction(DrugInteractionEntry {
        perpetrator: DrugId::from_name("paroxetine"),
        victim: DrugId::from_name("codeine"),
        mechanism: "CYP2D6 inhibition".into(),
        severity: Severity::Major,
        auc_fold_change: 0.3,
        clinical_note: "Reduced morphine conversion".into(),
    });
    db.add_interaction(DrugInteractionEntry {
        perpetrator: DrugId::from_name("omeprazole"),
        victim: DrugId::from_name("clopidogrel"),
        mechanism: "CYP2C19 inhibition".into(),
        severity: Severity::Major,
        auc_fold_change: 0.5,
        clinical_note: "Reduced antiplatelet effect".into(),
    });
    db.add_interaction(DrugInteractionEntry {
        perpetrator: DrugId::from_name("ciprofloxacin"),
        victim: DrugId::from_name("theophylline"),
        mechanism: "CYP1A2 inhibition".into(),
        severity: Severity::Moderate,
        auc_fold_change: 1.8,
        clinical_note: "Monitor theophylline levels".into(),
    });
    db.add_interaction(DrugInteractionEntry {
        perpetrator: DrugId::from_name("fluoxetine"),
        victim: DrugId::from_name("metoprolol"),
        mechanism: "CYP2D6 inhibition".into(),
        severity: Severity::Moderate,
        auc_fold_change: 2.0,
        clinical_note: "Monitor HR and BP".into(),
    });
    db.add_interaction(DrugInteractionEntry {
        perpetrator: DrugId::from_name("erythromycin"),
        victim: DrugId::from_name("midazolam"),
        mechanism: "CYP3A4 inhibition".into(),
        severity: Severity::Major,
        auc_fold_change: 4.0,
        clinical_note: "Excessive sedation risk".into(),
    });
    db.add_interaction(DrugInteractionEntry {
        perpetrator: DrugId::from_name("amiodarone"),
        victim: DrugId::from_name("digoxin"),
        mechanism: "P-gp inhibition + renal CL reduction".into(),
        severity: Severity::Major,
        auc_fold_change: 2.0,
        clinical_note: "Reduce digoxin dose by 50%".into(),
    });
    db.add_interaction(DrugInteractionEntry {
        perpetrator: DrugId::from_name("verapamil"),
        victim: DrugId::from_name("simvastatin"),
        mechanism: "CYP3A4 inhibition".into(),
        severity: Severity::Major,
        auc_fold_change: 2.5,
        clinical_note: "Limit simvastatin to 20mg".into(),
    });

    db
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_default_database() {
        let db = build_default_database();
        assert!(db.drug_count() >= 30);
        assert!(db.interactions.len() >= 10);
    }

    #[test]
    fn test_get_drug() {
        let db = build_default_database();
        let w = db.get_drug_by_name("warfarin").unwrap();
        assert!(w.half_life > 20.0);
        assert!(w.protein_binding > 0.9);
    }

    #[test]
    fn test_drugs_metabolized_by() {
        let db = build_default_database();
        let cyp3a4_drugs = db.drugs_metabolized_by(CypEnzyme::CYP3A4);
        assert!(cyp3a4_drugs.len() >= 8);
    }

    #[test]
    fn test_drugs_inhibiting() {
        let db = build_default_database();
        let inhibitors = db.drugs_inhibiting(CypEnzyme::CYP3A4);
        assert!(inhibitors.len() >= 3);
    }

    #[test]
    fn test_drugs_sharing_enzyme() {
        let db = build_default_database();
        let shared = db.drugs_sharing_enzyme(&DrugId::from_name("warfarin"));
        assert!(!shared.is_empty());
    }

    #[test]
    fn test_interaction_pairs() {
        let db = build_default_database();
        let drugs = vec![
            DrugId::from_name("fluconazole"),
            DrugId::from_name("warfarin"),
        ];
        let ints = db.get_interaction_pairs(&drugs);
        assert_eq!(ints.len(), 1);
    }

    #[test]
    fn test_drug_pk_calculations() {
        let db = build_default_database();
        let d = db.get_drug_by_name("warfarin").unwrap();
        let ke = d.elimination_rate_constant();
        assert!(ke > 0.0);
        let css = d.predicted_css_avg();
        assert!(css > 0.0);
    }

    #[test]
    fn test_primary_cyp() {
        let db = build_default_database();
        let w = db.get_drug_by_name("warfarin").unwrap();
        assert_eq!(w.primary_cyp(), Some(CypEnzyme::CYP2C9));
        let m = db.get_drug_by_name("midazolam").unwrap();
        assert_eq!(m.primary_cyp(), Some(CypEnzyme::CYP3A4));
    }

    #[test]
    fn test_drugs_by_class() {
        let db = build_default_database();
        let statins = db.drugs_by_class(DrugClass::Statin);
        assert!(statins.len() >= 2);
    }

    #[test]
    fn test_therapeutic_window() {
        let db = build_default_database();
        let d = db.get_drug_by_name("theophylline").unwrap();
        assert!(d.is_within_therapeutic_window(10.0));
        assert!(!d.is_within_therapeutic_window(100.0));
    }

    #[test]
    fn test_display() {
        let db = build_default_database();
        let d = db.get_drug_by_name("warfarin").unwrap();
        let s = format!("{}", d);
        assert!(s.contains("warfarin"));
    }

    #[test]
    fn test_inducing_drugs() {
        let db = build_default_database();
        let inducers = db.drugs_inducing(CypEnzyme::CYP3A4);
        assert!(inducers.len() >= 2);
    }
}
