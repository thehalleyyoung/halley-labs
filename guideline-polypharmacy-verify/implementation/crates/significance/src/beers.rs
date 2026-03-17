//! Beers Criteria 2023 (AGS) for potentially inappropriate medications in older adults.
//!
//! Implements screening against the five Beers Criteria tables:
//! 1. PIMs to avoid in most older adults
//! 2. PIMs to avoid due to specific disease/syndrome
//! 3. PIMs to use with caution
//! 4. Drug–drug interactions to avoid
//! 5. Medications requiring renal dose adjustment

use serde::{Deserialize, Serialize};
use std::fmt;

use crate::{Condition, Medication, PatientProfile};
use crate::RenalFunction;

// ─────────────────────────── Enums ───────────────────────────────────────

/// The five Beers Criteria tables.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BeersCategory {
    /// Table 2: PIMs to avoid in most older adults.
    AvoidInOlderAdults,
    /// Table 3: PIMs to avoid due to specific disease or syndrome.
    AvoidDueToCondition,
    /// Table 4: PIMs to use with caution.
    UseWithCaution,
    /// Table 5: Clinically important drug–drug interactions to avoid.
    AvoidDrugInteraction,
    /// Table 6: Medications requiring renal dose adjustment.
    DoseAdjustmentForRenal,
}

impl fmt::Display for BeersCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BeersCategory::AvoidInOlderAdults => write!(f, "Avoid in Older Adults"),
            BeersCategory::AvoidDueToCondition => write!(f, "Avoid Due to Condition"),
            BeersCategory::UseWithCaution => write!(f, "Use with Caution"),
            BeersCategory::AvoidDrugInteraction => write!(f, "Avoid Drug Interaction"),
            BeersCategory::DoseAdjustmentForRenal => write!(f, "Renal Dose Adjustment"),
        }
    }
}

/// Quality of evidence rating.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub enum QualityOfEvidence {
    VeryLow,
    Low,
    Moderate,
    High,
}

impl QualityOfEvidence {
    pub fn weight(&self) -> f64 {
        match self {
            QualityOfEvidence::VeryLow => 0.25,
            QualityOfEvidence::Low => 0.50,
            QualityOfEvidence::Moderate => 0.75,
            QualityOfEvidence::High => 1.0,
        }
    }
}

impl fmt::Display for QualityOfEvidence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QualityOfEvidence::VeryLow => write!(f, "Very Low"),
            QualityOfEvidence::Low => write!(f, "Low"),
            QualityOfEvidence::Moderate => write!(f, "Moderate"),
            QualityOfEvidence::High => write!(f, "High"),
        }
    }
}

/// Strength of recommendation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub enum StrengthOfRecommendation {
    Weak,
    Strong,
}

impl StrengthOfRecommendation {
    pub fn weight(&self) -> f64 {
        match self {
            StrengthOfRecommendation::Weak => 0.5,
            StrengthOfRecommendation::Strong => 1.0,
        }
    }
}

impl fmt::Display for StrengthOfRecommendation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StrengthOfRecommendation::Weak => write!(f, "Weak"),
            StrengthOfRecommendation::Strong => write!(f, "Strong"),
        }
    }
}

// ─────────────────────────── Structs ─────────────────────────────────────

/// A single Beers Criteria entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeersCriterion {
    pub id: String,
    pub category: BeersCategory,
    /// Medication names or classes this criterion applies to.
    pub medications: Vec<String>,
    /// Disease/condition context (for Table 3) or interacting drug (for Table 5).
    pub condition_context: Option<String>,
    pub recommendation: String,
    pub rationale: String,
    pub quality_of_evidence: QualityOfEvidence,
    pub strength_of_recommendation: StrengthOfRecommendation,
}

impl BeersCriterion {
    pub fn new(
        id: &str,
        category: BeersCategory,
        medications: &[&str],
        recommendation: &str,
        rationale: &str,
        quality: QualityOfEvidence,
        strength: StrengthOfRecommendation,
    ) -> Self {
        BeersCriterion {
            id: id.to_string(),
            category,
            medications: medications.iter().map(|s| s.to_lowercase()).collect(),
            condition_context: None,
            recommendation: recommendation.to_string(),
            rationale: rationale.to_string(),
            quality_of_evidence: quality,
            strength_of_recommendation: strength,
        }
    }

    pub fn with_condition(mut self, condition: &str) -> Self {
        self.condition_context = Some(condition.to_string());
        self
    }

    /// Composite weight: quality × strength.
    pub fn composite_weight(&self) -> f64 {
        self.quality_of_evidence.weight() * self.strength_of_recommendation.weight()
    }

    /// Whether a given medication name matches this criterion.
    pub fn matches_medication(&self, med_name: &str) -> bool {
        let lower = med_name.to_lowercase();
        self.medications.iter().any(|m| {
            lower.contains(m) || m.contains(&lower)
        })
    }

    /// Whether a given drug class matches this criterion's listed medications/classes.
    pub fn matches_class(&self, drug_class: &str) -> bool {
        let lower = drug_class.to_lowercase();
        self.medications.iter().any(|m| {
            lower.contains(m) || m.contains(&lower)
        })
    }
}

impl fmt::Display for BeersCriterion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}: {}", self.id, self.category, self.recommendation)
    }
}

/// A violation discovered when checking a patient against Beers Criteria.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeersViolation {
    pub criterion_id: String,
    pub category: BeersCategory,
    pub violated_medication: String,
    pub interacting_medication: Option<String>,
    pub reason: String,
    pub severity_score: f64,
    pub recommendation: String,
    pub quality_of_evidence: QualityOfEvidence,
    pub strength_of_recommendation: StrengthOfRecommendation,
}

impl BeersViolation {
    pub fn new(
        criterion: &BeersCriterion,
        violated_med: &str,
        reason: &str,
    ) -> Self {
        BeersViolation {
            criterion_id: criterion.id.clone(),
            category: criterion.category,
            violated_medication: violated_med.to_string(),
            interacting_medication: None,
            reason: reason.to_string(),
            severity_score: criterion.composite_weight(),
            recommendation: criterion.recommendation.clone(),
            quality_of_evidence: criterion.quality_of_evidence,
            strength_of_recommendation: criterion.strength_of_recommendation,
        }
    }

    pub fn with_interacting_med(mut self, med: &str) -> Self {
        self.interacting_medication = Some(med.to_string());
        self
    }

    pub fn is_strong_recommendation(&self) -> bool {
        self.strength_of_recommendation == StrengthOfRecommendation::Strong
    }
}

impl fmt::Display for BeersViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] {} — {} (score: {:.2})",
            self.criterion_id, self.violated_medication, self.reason, self.severity_score,
        )
    }
}

// ─────────────────────────── BeersCriteria Database ──────────────────────

/// In-memory Beers Criteria 2023 database.
#[derive(Debug, Clone)]
pub struct BeersCriteria {
    criteria: Vec<BeersCriterion>,
}

impl BeersCriteria {
    pub fn empty() -> Self {
        BeersCriteria { criteria: Vec::new() }
    }

    /// Create the database pre-populated with 2023 AGS Beers Criteria.
    pub fn with_defaults() -> Self {
        let mut bc = Self::empty();
        bc.load_defaults();
        bc
    }

    pub fn add(&mut self, criterion: BeersCriterion) {
        self.criteria.push(criterion);
    }

    pub fn len(&self) -> usize {
        self.criteria.len()
    }

    pub fn is_empty(&self) -> bool {
        self.criteria.is_empty()
    }

    pub fn criteria(&self) -> &[BeersCriterion] {
        &self.criteria
    }

    pub fn get_by_id(&self, id: &str) -> Option<&BeersCriterion> {
        self.criteria.iter().find(|c| c.id == id)
    }

    pub fn get_by_category(&self, category: BeersCategory) -> Vec<&BeersCriterion> {
        self.criteria.iter().filter(|c| c.category == category).collect()
    }

    // ── Patient-level checks ────────────────────────────────────────────

    /// Check all criteria for a patient and their medications.
    pub fn check_patient(
        &self,
        patient: &PatientProfile,
        medications: &[Medication],
    ) -> Vec<BeersViolation> {
        let mut violations = Vec::new();

        if !patient.is_elderly() {
            return violations;
        }

        // Table 2: PIMs to avoid in older adults
        for med in medications {
            violations.extend(self.check_single_medication(med, patient));
        }

        // Table 5: Drug–drug interactions
        for i in 0..medications.len() {
            for j in (i + 1)..medications.len() {
                violations.extend(self.check_drug_interaction(
                    &medications[i],
                    &medications[j],
                    patient,
                ));
            }
        }

        // Table 6: Renal dose adjustments
        for med in medications {
            violations.extend(self.check_renal_dosing(med, patient));
        }

        violations
    }

    /// Check a single medication against Table 2 (avoid) and Table 3 (condition-specific).
    pub fn check_single_medication(
        &self,
        med: &Medication,
        patient: &PatientProfile,
    ) -> Vec<BeersViolation> {
        let mut violations = Vec::new();

        for criterion in &self.criteria {
            match criterion.category {
                BeersCategory::AvoidInOlderAdults => {
                    if criterion.matches_medication(&med.name)
                        || criterion.matches_class(&med.drug_class)
                    {
                        violations.push(BeersViolation::new(
                            criterion,
                            &med.name,
                            &format!(
                                "PIM: {} — {}",
                                criterion.rationale, criterion.recommendation
                            ),
                        ));
                    }
                }
                BeersCategory::AvoidDueToCondition => {
                    if (criterion.matches_medication(&med.name)
                        || criterion.matches_class(&med.drug_class))
                        && condition_matches(
                            criterion.condition_context.as_deref(),
                            &patient.conditions,
                        )
                    {
                        violations.push(BeersViolation::new(
                            criterion,
                            &med.name,
                            &format!(
                                "Avoid due to condition ({}): {}",
                                criterion.condition_context.as_deref().unwrap_or(""),
                                criterion.rationale,
                            ),
                        ));
                    }
                }
                BeersCategory::UseWithCaution => {
                    if criterion.matches_medication(&med.name)
                        || criterion.matches_class(&med.drug_class)
                    {
                        violations.push(BeersViolation::new(
                            criterion,
                            &med.name,
                            &format!("Use with caution: {}", criterion.rationale),
                        ));
                    }
                }
                _ => {}
            }
        }

        violations
    }

    /// Check two medications against Table 5 (drug–drug interactions to avoid).
    pub fn check_drug_interaction(
        &self,
        med_a: &Medication,
        med_b: &Medication,
        _patient: &PatientProfile,
    ) -> Vec<BeersViolation> {
        let mut violations = Vec::new();

        for criterion in self.get_by_category(BeersCategory::AvoidDrugInteraction) {
            let meds = &criterion.medications;
            if meds.len() >= 2 {
                let matches_a_first = medication_in_list(&med_a.name, &med_a.drug_class, &meds[..meds.len() / 2])
                    && medication_in_list(&med_b.name, &med_b.drug_class, &meds[meds.len() / 2..]);
                let matches_b_first = medication_in_list(&med_b.name, &med_b.drug_class, &meds[..meds.len() / 2])
                    && medication_in_list(&med_a.name, &med_a.drug_class, &meds[meds.len() / 2..]);

                if matches_a_first || matches_b_first {
                    let v = BeersViolation::new(
                        criterion,
                        &med_a.name,
                        &format!(
                            "Drug interaction to avoid: {} + {} — {}",
                            med_a.name, med_b.name, criterion.rationale,
                        ),
                    ).with_interacting_med(&med_b.name);
                    violations.push(v);
                }
            }
        }

        violations
    }

    /// Check a medication against Table 6 (renal dose adjustment).
    fn check_renal_dosing(
        &self,
        med: &Medication,
        patient: &PatientProfile,
    ) -> Vec<BeersViolation> {
        let mut violations = Vec::new();
        let renal = patient.renal_function();

        for criterion in self.get_by_category(BeersCategory::DoseAdjustmentForRenal) {
            if criterion.matches_medication(&med.name) || criterion.matches_class(&med.drug_class)
            {
                let needs_adjustment = match renal {
                    RenalFunction::Normal | RenalFunction::Mild => false,
                    RenalFunction::Moderate => true,
                    RenalFunction::Severe | RenalFunction::EndStage => true,
                };

                if needs_adjustment {
                    violations.push(BeersViolation::new(
                        criterion,
                        &med.name,
                        &format!(
                            "Renal dose adjustment needed (eGFR-based renal function: {:?}): {}",
                            renal, criterion.recommendation,
                        ),
                    ));
                }
            }
        }

        violations
    }

    // ── Default data ────────────────────────────────────────────────────

    fn load_defaults(&mut self) {
        // ═══════════════════════════════════════════════════════════════
        // TABLE 2 — PIMs to Avoid in Most Older Adults
        // ═══════════════════════════════════════════════════════════════

        self.add(BeersCriterion::new(
            "BEERS-2-01", BeersCategory::AvoidInOlderAdults,
            &["chlordiazepoxide", "diazepam", "flurazepam", "clonazepam", "clorazepate"],
            "Avoid long-acting benzodiazepines",
            "Older adults have increased sensitivity; prolonged sedation, falls, fractures, cognitive impairment",
            QualityOfEvidence::Moderate,
            StrengthOfRecommendation::Strong,
        ));

        self.add(BeersCriterion::new(
            "BEERS-2-02", BeersCategory::AvoidInOlderAdults,
            &["alprazolam", "lorazepam", "oxazepam", "temazepam", "triazolam"],
            "Avoid short- and intermediate-acting benzodiazepines",
            "Increased risk of cognitive impairment, delirium, falls, fractures, and motor vehicle crashes",
            QualityOfEvidence::Moderate,
            StrengthOfRecommendation::Strong,
        ));

        self.add(BeersCriterion::new(
            "BEERS-2-03", BeersCategory::AvoidInOlderAdults,
            &["diphenhydramine", "chlorpheniramine", "hydroxyzine", "promethazine", "brompheniramine", "cyproheptadine", "dexchlorpheniramine"],
            "Avoid first-generation antihistamines",
            "Highly anticholinergic; risk of confusion, dry mouth, constipation, urinary retention",
            QualityOfEvidence::Moderate,
            StrengthOfRecommendation::Strong,
        ));

        self.add(BeersCriterion::new(
            "BEERS-2-04", BeersCategory::AvoidInOlderAdults,
            &["amitriptyline", "clomipramine", "doxepin", "imipramine", "nortriptyline", "trimipramine"],
            "Avoid tertiary tricyclic antidepressants",
            "Highly anticholinergic, sedating, and cause orthostatic hypotension; risk of falls and cardiac conduction disturbance",
            QualityOfEvidence::High,
            StrengthOfRecommendation::Strong,
        ));

        self.add(BeersCriterion::new(
            "BEERS-2-05", BeersCategory::AvoidInOlderAdults,
            &["meperidine"],
            "Avoid meperidine",
            "Neurotoxic metabolite normeperidine accumulates; risk of seizures, delirium",
            QualityOfEvidence::High,
            StrengthOfRecommendation::Strong,
        ));

        self.add(BeersCriterion::new(
            "BEERS-2-06", BeersCategory::AvoidInOlderAdults,
            &["indomethacin", "ketorolac"],
            "Avoid indomethacin and ketorolac",
            "Highest risk of GI bleeding and renal injury among NSAIDs",
            QualityOfEvidence::Moderate,
            StrengthOfRecommendation::Strong,
        ));

        self.add(BeersCriterion::new(
            "BEERS-2-07", BeersCategory::AvoidInOlderAdults,
            &["glyburide", "glimepiride", "chlorpropamide"],
            "Avoid long-acting sulfonylureas",
            "Higher risk of severe prolonged hypoglycemia in older adults",
            QualityOfEvidence::High,
            StrengthOfRecommendation::Strong,
        ));

        self.add(BeersCriterion::new(
            "BEERS-2-08", BeersCategory::AvoidInOlderAdults,
            &["metoclopramide"],
            "Avoid metoclopramide unless for gastroparesis",
            "Risk of extrapyramidal effects including tardive dyskinesia",
            QualityOfEvidence::Moderate,
            StrengthOfRecommendation::Strong,
        ));

        self.add(BeersCriterion::new(
            "BEERS-2-09", BeersCategory::AvoidInOlderAdults,
            &["proton pump inhibitor", "omeprazole", "esomeprazole", "lansoprazole", "pantoprazole", "rabeprazole"],
            "Avoid PPIs beyond 8 weeks without clear indication",
            "Risk of C. difficile infection, bone loss, fractures, hypomagnesemia, fundic gland polyps",
            QualityOfEvidence::High,
            StrengthOfRecommendation::Strong,
        ));

        self.add(BeersCriterion::new(
            "BEERS-2-10", BeersCategory::AvoidInOlderAdults,
            &["desmopressin"],
            "Avoid desmopressin for nocturia",
            "High risk of hyponatremia in older adults",
            QualityOfEvidence::Moderate,
            StrengthOfRecommendation::Strong,
        ));

        self.add(BeersCriterion::new(
            "BEERS-2-11", BeersCategory::AvoidInOlderAdults,
            &["nitrofurantoin"],
            "Avoid nitrofurantoin for long-term suppression or if CrCl <30",
            "Potential for pulmonary toxicity, hepatotoxicity, peripheral neuropathy; lacks efficacy at low CrCl",
            QualityOfEvidence::Low,
            StrengthOfRecommendation::Strong,
        ));

        self.add(BeersCriterion::new(
            "BEERS-2-12", BeersCategory::AvoidInOlderAdults,
            &["sliding scale insulin"],
            "Avoid sliding-scale insulin regimens without basal",
            "Higher risk of hypoglycemia without improved glycemic management",
            QualityOfEvidence::Moderate,
            StrengthOfRecommendation::Strong,
        ));

        self.add(BeersCriterion::new(
            "BEERS-2-13", BeersCategory::AvoidInOlderAdults,
            &["nifedipine"],
            "Avoid short-acting nifedipine (immediate release)",
            "Risk of hypotension and myocardial ischemia",
            QualityOfEvidence::High,
            StrengthOfRecommendation::Strong,
        ));

        self.add(BeersCriterion::new(
            "BEERS-2-14", BeersCategory::AvoidInOlderAdults,
            &["doxazosin", "prazosin", "terazosin"],
            "Avoid alpha-1 blockers for hypertension",
            "High risk of orthostatic hypotension; not recommended as routine treatment for hypertension",
            QualityOfEvidence::Moderate,
            StrengthOfRecommendation::Strong,
        ));

        self.add(BeersCriterion::new(
            "BEERS-2-15", BeersCategory::AvoidInOlderAdults,
            &["methyldopa", "reserpine", "guanfacine", "clonidine"],
            "Avoid centrally-acting alpha-agonists for hypertension",
            "High risk of CNS adverse effects; risk of bradycardia and orthostatic hypotension",
            QualityOfEvidence::Low,
            StrengthOfRecommendation::Strong,
        ));

        self.add(BeersCriterion::new(
            "BEERS-2-16", BeersCategory::AvoidInOlderAdults,
            &["mineral oil"],
            "Avoid mineral oil (oral)",
            "Risk of aspiration pneumonia",
            QualityOfEvidence::Moderate,
            StrengthOfRecommendation::Strong,
        ));

        self.add(BeersCriterion::new(
            "BEERS-2-17", BeersCategory::AvoidInOlderAdults,
            &["carisoprodol", "chlorzoxazone", "cyclobenzaprine", "metaxalone", "methocarbamol", "orphenadrine"],
            "Avoid skeletal muscle relaxants",
            "Poorly tolerated due to anticholinergic effects, sedation, and fracture risk",
            QualityOfEvidence::Moderate,
            StrengthOfRecommendation::Strong,
        ));

        self.add(BeersCriterion::new(
            "BEERS-2-18", BeersCategory::AvoidInOlderAdults,
            &["trimethobenzamide"],
            "Avoid trimethobenzamide",
            "One of the least effective antiemetics; can cause extrapyramidal side effects",
            QualityOfEvidence::Moderate,
            StrengthOfRecommendation::Strong,
        ));

        // ═══════════════════════════════════════════════════════════════
        // TABLE 3 — PIMs Due to Disease/Syndrome Interaction
        // ═══════════════════════════════════════════════════════════════

        self.add(BeersCriterion::new(
            "BEERS-3-01", BeersCategory::AvoidDueToCondition,
            &["anticholinergic", "diphenhydramine", "oxybutynin", "tolterodine", "benztropine", "trihexyphenidyl"],
            "Avoid anticholinergics in dementia or cognitive impairment",
            "Worsen cognitive decline; increased risk of delirium",
            QualityOfEvidence::Moderate,
            StrengthOfRecommendation::Strong,
        ).with_condition("dementia"));

        self.add(BeersCriterion::new(
            "BEERS-3-02", BeersCategory::AvoidDueToCondition,
            &["antipsychotic", "haloperidol", "risperidone", "olanzapine", "quetiapine", "aripiprazole"],
            "Avoid antipsychotics in dementia unless for documented psychosis or agitation causing danger",
            "Increased mortality (black box warning) and cerebrovascular events",
            QualityOfEvidence::Moderate,
            StrengthOfRecommendation::Strong,
        ).with_condition("dementia"));

        self.add(BeersCriterion::new(
            "BEERS-3-03", BeersCategory::AvoidDueToCondition,
            &["nsaid", "ibuprofen", "naproxen", "diclofenac", "meloxicam", "celecoxib", "piroxicam", "indomethacin"],
            "Avoid NSAIDs in chronic kidney disease (stage IV or higher)",
            "May worsen renal function; increase risk of fluid retention and hyperkalemia",
            QualityOfEvidence::Moderate,
            StrengthOfRecommendation::Strong,
        ).with_condition("chronic kidney disease"));

        self.add(BeersCriterion::new(
            "BEERS-3-04", BeersCategory::AvoidDueToCondition,
            &["nsaid", "ibuprofen", "naproxen", "aspirin", "diclofenac"],
            "Avoid NSAIDs in heart failure",
            "Promote fluid retention and exacerbate heart failure",
            QualityOfEvidence::Moderate,
            StrengthOfRecommendation::Strong,
        ).with_condition("heart failure"));

        self.add(BeersCriterion::new(
            "BEERS-3-05", BeersCategory::AvoidDueToCondition,
            &["nsaid", "corticosteroid", "prednisone", "dexamethasone"],
            "Avoid NSAIDs and corticosteroids without gastroprotection in GI ulcer history",
            "High risk of GI bleeding recurrence",
            QualityOfEvidence::Moderate,
            StrengthOfRecommendation::Strong,
        ).with_condition("peptic ulcer"));

        self.add(BeersCriterion::new(
            "BEERS-3-06", BeersCategory::AvoidDueToCondition,
            &["thiazolidinedione", "pioglitazone", "rosiglitazone"],
            "Avoid thiazolidinediones in heart failure",
            "Cause fluid retention; may exacerbate or cause heart failure",
            QualityOfEvidence::High,
            StrengthOfRecommendation::Strong,
        ).with_condition("heart failure"));

        self.add(BeersCriterion::new(
            "BEERS-3-07", BeersCategory::AvoidDueToCondition,
            &["diltiazem", "verapamil"],
            "Avoid non-dihydropyridine calcium channel blockers in heart failure with reduced EF",
            "Negative inotropic effects may worsen heart failure",
            QualityOfEvidence::Moderate,
            StrengthOfRecommendation::Strong,
        ).with_condition("heart failure"));

        self.add(BeersCriterion::new(
            "BEERS-3-08", BeersCategory::AvoidDueToCondition,
            &["anticholinergic", "oxybutynin", "tolterodine", "solifenacin", "fesoterodine"],
            "Avoid anticholinergics in benign prostatic hyperplasia / urinary retention",
            "May decrease urinary flow and cause acute urinary retention",
            QualityOfEvidence::Moderate,
            StrengthOfRecommendation::Strong,
        ).with_condition("benign prostatic hyperplasia"));

        self.add(BeersCriterion::new(
            "BEERS-3-09", BeersCategory::AvoidDueToCondition,
            &["benzodiazepine", "diazepam", "lorazepam", "alprazolam", "clonazepam"],
            "Avoid benzodiazepines in delirium",
            "May worsen delirium; consider underlying cause; exception for alcohol/BZD withdrawal",
            QualityOfEvidence::Moderate,
            StrengthOfRecommendation::Strong,
        ).with_condition("delirium"));

        self.add(BeersCriterion::new(
            "BEERS-3-10", BeersCategory::AvoidDueToCondition,
            &["metformin"],
            "Avoid metformin if eGFR <30",
            "Increased risk of lactic acidosis with severe renal impairment",
            QualityOfEvidence::Moderate,
            StrengthOfRecommendation::Strong,
        ).with_condition("severe renal impairment"));

        self.add(BeersCriterion::new(
            "BEERS-3-11", BeersCategory::AvoidDueToCondition,
            &["seizure threshold lowering", "bupropion", "tramadol", "theophylline"],
            "Avoid seizure-threshold-lowering drugs in epilepsy",
            "May precipitate seizures",
            QualityOfEvidence::Moderate,
            StrengthOfRecommendation::Strong,
        ).with_condition("epilepsy"));

        self.add(BeersCriterion::new(
            "BEERS-3-12", BeersCategory::AvoidDueToCondition,
            &["anticholinergic", "antihistamine", "antispasmodic"],
            "Avoid strongly anticholinergic drugs in constipation",
            "May worsen constipation",
            QualityOfEvidence::Low,
            StrengthOfRecommendation::Weak,
        ).with_condition("chronic constipation"));

        self.add(BeersCriterion::new(
            "BEERS-3-13", BeersCategory::AvoidDueToCondition,
            &["alpha-blocker", "doxazosin", "prazosin", "terazosin"],
            "Avoid alpha-blockers in syncope",
            "Increase risk of orthostatic hypotension and recurrent syncope",
            QualityOfEvidence::Moderate,
            StrengthOfRecommendation::Strong,
        ).with_condition("syncope"));

        self.add(BeersCriterion::new(
            "BEERS-3-14", BeersCategory::AvoidDueToCondition,
            &["ssri", "snri", "fluoxetine", "sertraline", "paroxetine", "citalopram", "escitalopram", "venlafaxine", "duloxetine"],
            "Use SSRIs/SNRIs with caution in fall history",
            "Increase risk of falls and fractures; SIADH-related hyponatremia",
            QualityOfEvidence::Moderate,
            StrengthOfRecommendation::Strong,
        ).with_condition("falls"));

        // ═══════════════════════════════════════════════════════════════
        // TABLE 4 — PIMs to Use with Caution
        // ═══════════════════════════════════════════════════════════════

        self.add(BeersCriterion::new(
            "BEERS-4-01", BeersCategory::UseWithCaution,
            &["aspirin"],
            "Use aspirin for primary prevention with caution in adults ≥70",
            "Risk of major bleeding may outweigh cardiovascular benefit in primary prevention",
            QualityOfEvidence::Moderate,
            StrengthOfRecommendation::Weak,
        ));

        self.add(BeersCriterion::new(
            "BEERS-4-02", BeersCategory::UseWithCaution,
            &["dabigatran"],
            "Use dabigatran with caution in adults ≥75",
            "Increased risk of GI bleeding compared to warfarin in older adults",
            QualityOfEvidence::Moderate,
            StrengthOfRecommendation::Weak,
        ));

        self.add(BeersCriterion::new(
            "BEERS-4-03", BeersCategory::UseWithCaution,
            &["prasugrel"],
            "Avoid prasugrel in adults ≥75",
            "Increased risk of fatal and intracranial bleeding",
            QualityOfEvidence::Moderate,
            StrengthOfRecommendation::Strong,
        ));

        self.add(BeersCriterion::new(
            "BEERS-4-04", BeersCategory::UseWithCaution,
            &["carbamazepine", "oxcarbazepine"],
            "Use carbamazepine and oxcarbazepine with caution",
            "Can cause SIADH, hyponatremia; ataxia, impaired gait",
            QualityOfEvidence::Moderate,
            StrengthOfRecommendation::Weak,
        ));

        self.add(BeersCriterion::new(
            "BEERS-4-05", BeersCategory::UseWithCaution,
            &["mirtazapine"],
            "Use mirtazapine with caution",
            "May cause or exacerbate SIADH/hyponatremia; weight gain",
            QualityOfEvidence::Low,
            StrengthOfRecommendation::Weak,
        ));

        self.add(BeersCriterion::new(
            "BEERS-4-06", BeersCategory::UseWithCaution,
            &["diuretic", "hydrochlorothiazide", "furosemide", "chlorthalidone"],
            "Use diuretics with caution",
            "May exacerbate or cause SIADH/hyponatremia; monitor sodium",
            QualityOfEvidence::Moderate,
            StrengthOfRecommendation::Weak,
        ));

        self.add(BeersCriterion::new(
            "BEERS-4-07", BeersCategory::UseWithCaution,
            &["tramadol"],
            "Use tramadol with caution",
            "Serotonin syndrome risk when combined with other serotonergic agents; lowers seizure threshold; risk of hyponatremia",
            QualityOfEvidence::Moderate,
            StrengthOfRecommendation::Weak,
        ));

        // ═══════════════════════════════════════════════════════════════
        // TABLE 5 — Drug–Drug Interactions to Avoid
        // ═══════════════════════════════════════════════════════════════

        self.add(BeersCriterion::new(
            "BEERS-5-01", BeersCategory::AvoidDrugInteraction,
            &["opioid", "benzodiazepine"],
            "Avoid combining opioids with benzodiazepines",
            "Increased risk of overdose, respiratory depression, and death (FDA Black Box Warning)",
            QualityOfEvidence::Moderate,
            StrengthOfRecommendation::Strong,
        ));

        self.add(BeersCriterion::new(
            "BEERS-5-02", BeersCategory::AvoidDrugInteraction,
            &["opioid", "gabapentin", "pregabalin"],
            "Avoid combining opioids with gabapentinoids",
            "Increased risk of severe respiratory depression, sedation, and death",
            QualityOfEvidence::Moderate,
            StrengthOfRecommendation::Strong,
        ));

        self.add(BeersCriterion::new(
            "BEERS-5-03", BeersCategory::AvoidDrugInteraction,
            &["ace inhibitor", "lisinopril", "enalapril", "potassium-sparing diuretic", "spironolactone", "amiloride", "triamterene"],
            "Avoid ACE inhibitor/ARB + potassium-sparing diuretic without monitoring",
            "Increased risk of hyperkalemia",
            QualityOfEvidence::Moderate,
            StrengthOfRecommendation::Strong,
        ));

        self.add(BeersCriterion::new(
            "BEERS-5-04", BeersCategory::AvoidDrugInteraction,
            &["lithium", "ace inhibitor", "lisinopril", "enalapril"],
            "Avoid lithium with ACE inhibitors",
            "Increased risk of lithium toxicity due to reduced renal clearance",
            QualityOfEvidence::Moderate,
            StrengthOfRecommendation::Strong,
        ));

        self.add(BeersCriterion::new(
            "BEERS-5-05", BeersCategory::AvoidDrugInteraction,
            &["warfarin", "nsaid", "ibuprofen", "naproxen", "diclofenac"],
            "Avoid combining warfarin with NSAIDs",
            "Increases risk of GI bleeding",
            QualityOfEvidence::High,
            StrengthOfRecommendation::Strong,
        ));

        self.add(BeersCriterion::new(
            "BEERS-5-06", BeersCategory::AvoidDrugInteraction,
            &["anticholinergic", "cholinesterase inhibitor", "donepezil", "rivastigmine", "galantamine"],
            "Avoid anticholinergics with cholinesterase inhibitors",
            "Pharmacodynamic antagonism; reduces efficacy of cholinesterase inhibitor",
            QualityOfEvidence::Moderate,
            StrengthOfRecommendation::Strong,
        ));

        self.add(BeersCriterion::new(
            "BEERS-5-07", BeersCategory::AvoidDrugInteraction,
            &["theophylline", "cimetidine"],
            "Avoid combining theophylline with cimetidine",
            "Cimetidine inhibits theophylline metabolism, increasing toxicity risk",
            QualityOfEvidence::Moderate,
            StrengthOfRecommendation::Strong,
        ));

        self.add(BeersCriterion::new(
            "BEERS-5-08", BeersCategory::AvoidDrugInteraction,
            &["corticosteroid", "nsaid"],
            "Avoid corticosteroid + NSAID without gastroprotection",
            "Greatly increased risk of GI ulceration and bleeding",
            QualityOfEvidence::Moderate,
            StrengthOfRecommendation::Strong,
        ));

        self.add(BeersCriterion::new(
            "BEERS-5-09", BeersCategory::AvoidDrugInteraction,
            &["ssri", "snri", "fluoxetine", "sertraline", "tramadol", "meperidine"],
            "Avoid SSRI/SNRI + serotonergic opioid",
            "Increased risk of serotonin syndrome",
            QualityOfEvidence::Moderate,
            StrengthOfRecommendation::Strong,
        ));

        self.add(BeersCriterion::new(
            "BEERS-5-10", BeersCategory::AvoidDrugInteraction,
            &["antiplatelet", "anticoagulant", "warfarin", "aspirin", "clopidogrel"],
            "Avoid triple antithrombotic therapy (2 antiplatelets + anticoagulant) when possible",
            "Significantly increased bleeding risk; use shortest duration if necessary",
            QualityOfEvidence::Moderate,
            StrengthOfRecommendation::Strong,
        ));

        // ═══════════════════════════════════════════════════════════════
        // TABLE 6 — Renal Dose Adjustments
        // ═══════════════════════════════════════════════════════════════

        self.add(BeersCriterion::new(
            "BEERS-6-01", BeersCategory::DoseAdjustmentForRenal,
            &["apixaban"],
            "Reduce apixaban dose to 2.5 mg BID if CrCl 15-25 mL/min; avoid if <15",
            "Accumulation increases bleeding risk in renal impairment",
            QualityOfEvidence::Moderate,
            StrengthOfRecommendation::Strong,
        ));

        self.add(BeersCriterion::new(
            "BEERS-6-02", BeersCategory::DoseAdjustmentForRenal,
            &["rivaroxaban"],
            "Reduce rivaroxaban dose if CrCl 15-50 mL/min; avoid if <15",
            "Renal clearance constitutes ~33% of total clearance",
            QualityOfEvidence::Moderate,
            StrengthOfRecommendation::Strong,
        ));

        self.add(BeersCriterion::new(
            "BEERS-6-03", BeersCategory::DoseAdjustmentForRenal,
            &["dabigatran"],
            "Reduce dabigatran dose if CrCl 30-50 mL/min; avoid if <30",
            "80% renally cleared; accumulation risk with renal impairment",
            QualityOfEvidence::Moderate,
            StrengthOfRecommendation::Strong,
        ));

        self.add(BeersCriterion::new(
            "BEERS-6-04", BeersCategory::DoseAdjustmentForRenal,
            &["edoxaban"],
            "Reduce edoxaban to 30 mg if CrCl 15-50 mL/min; avoid if <15 or >95",
            "Renal impairment increases exposure; paradoxically less effective at high CrCl",
            QualityOfEvidence::Moderate,
            StrengthOfRecommendation::Strong,
        ));

        self.add(BeersCriterion::new(
            "BEERS-6-05", BeersCategory::DoseAdjustmentForRenal,
            &["gabapentin"],
            "Reduce gabapentin dose: 300 mg daily if CrCl <15; adjust proportionally for CrCl 15-60",
            "Exclusively renally cleared; accumulation causes sedation and falls",
            QualityOfEvidence::Moderate,
            StrengthOfRecommendation::Strong,
        ));

        self.add(BeersCriterion::new(
            "BEERS-6-06", BeersCategory::DoseAdjustmentForRenal,
            &["pregabalin"],
            "Reduce pregabalin dose if CrCl <60; 25-75 mg daily if <15",
            "Renally cleared; neurotoxic effects at supratherapeutic levels",
            QualityOfEvidence::Moderate,
            StrengthOfRecommendation::Strong,
        ));

        self.add(BeersCriterion::new(
            "BEERS-6-07", BeersCategory::DoseAdjustmentForRenal,
            &["levetiracetam"],
            "Reduce levetiracetam dose if CrCl <80",
            "66% renally cleared; dose reduction based on CrCl",
            QualityOfEvidence::Moderate,
            StrengthOfRecommendation::Strong,
        ));

        self.add(BeersCriterion::new(
            "BEERS-6-08", BeersCategory::DoseAdjustmentForRenal,
            &["enoxaparin"],
            "Reduce enoxaparin to 1 mg/kg once daily if CrCl <30",
            "Renally cleared; accumulation increases major bleeding risk",
            QualityOfEvidence::Moderate,
            StrengthOfRecommendation::Strong,
        ));

        self.add(BeersCriterion::new(
            "BEERS-6-09", BeersCategory::DoseAdjustmentForRenal,
            &["colchicine"],
            "Reduce colchicine dose if CrCl <30; avoid concomitant CYP3A4/P-gp inhibitors",
            "Reduced clearance increases toxicity risk",
            QualityOfEvidence::Moderate,
            StrengthOfRecommendation::Strong,
        ));

        self.add(BeersCriterion::new(
            "BEERS-6-10", BeersCategory::DoseAdjustmentForRenal,
            &["metformin"],
            "Avoid metformin if eGFR <30; reduce dose if 30-45",
            "Risk of lactic acidosis with impaired renal clearance",
            QualityOfEvidence::Moderate,
            StrengthOfRecommendation::Strong,
        ));

        self.add(BeersCriterion::new(
            "BEERS-6-11", BeersCategory::DoseAdjustmentForRenal,
            &["duloxetine"],
            "Avoid duloxetine if CrCl <30",
            "Active metabolites accumulate with severe renal impairment",
            QualityOfEvidence::Moderate,
            StrengthOfRecommendation::Strong,
        ));

        self.add(BeersCriterion::new(
            "BEERS-6-12", BeersCategory::DoseAdjustmentForRenal,
            &["spironolactone"],
            "Avoid spironolactone if CrCl <30",
            "Severe hyperkalemia risk with impaired renal potassium excretion",
            QualityOfEvidence::Moderate,
            StrengthOfRecommendation::Strong,
        ));

        self.add(BeersCriterion::new(
            "BEERS-6-13", BeersCategory::DoseAdjustmentForRenal,
            &["allopurinol"],
            "Start allopurinol at 100 mg daily if CrCl <60; titrate slowly",
            "Accumulation of oxypurinol increases hypersensitivity risk",
            QualityOfEvidence::Moderate,
            StrengthOfRecommendation::Strong,
        ));

        self.add(BeersCriterion::new(
            "BEERS-6-14", BeersCategory::DoseAdjustmentForRenal,
            &["ranitidine", "famotidine", "cimetidine"],
            "Reduce H2-blocker dose if CrCl <50",
            "Renally cleared; CNS effects (confusion) at elevated levels",
            QualityOfEvidence::Low,
            StrengthOfRecommendation::Strong,
        ));

        self.add(BeersCriterion::new(
            "BEERS-6-15", BeersCategory::DoseAdjustmentForRenal,
            &["ciprofloxacin", "levofloxacin"],
            "Reduce fluoroquinolone dose if CrCl <30",
            "Renally cleared; CNS toxicity and tendon damage risk increase",
            QualityOfEvidence::Moderate,
            StrengthOfRecommendation::Strong,
        ));
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────

/// Check whether a patient's conditions match the criterion's condition context.
fn condition_matches(context: Option<&str>, conditions: &[Condition]) -> bool {
    let ctx = match context {
        Some(c) => c.to_lowercase(),
        None => return false,
    };

    for cond in conditions {
        if !cond.active {
            continue;
        }
        let lower_name = cond.name.to_lowercase();
        let lower_code = cond.code.to_lowercase();

        if lower_name.contains(&ctx) || ctx.contains(&lower_name) {
            return true;
        }

        // Map common condition contexts to ICD-10 prefixes
        let matched = match ctx.as_str() {
            "dementia" => lower_name.contains("dementia") || lower_name.contains("alzheimer") || lower_code.starts_with("f0") || lower_code.starts_with("g30"),
            "chronic kidney disease" => lower_name.contains("kidney") || lower_name.contains("renal") || lower_code.starts_with("n18"),
            "heart failure" => lower_name.contains("heart failure") || lower_code.starts_with("i50"),
            "peptic ulcer" => lower_name.contains("ulcer") || lower_code.starts_with("k25") || lower_code.starts_with("k26") || lower_code.starts_with("k27"),
            "benign prostatic hyperplasia" => lower_name.contains("prostat") || lower_code.starts_with("n40"),
            "delirium" => lower_name.contains("delirium") || lower_code.starts_with("f05"),
            "severe renal impairment" => lower_name.contains("kidney") || lower_name.contains("renal") || lower_code.starts_with("n18"),
            "epilepsy" => lower_name.contains("epilep") || lower_name.contains("seizure") || lower_code.starts_with("g40"),
            "chronic constipation" => lower_name.contains("constipat") || lower_code.starts_with("k59"),
            "syncope" => lower_name.contains("syncope") || lower_code.starts_with("r55"),
            "falls" => lower_name.contains("fall") || lower_code.starts_with("w"),
            _ => false,
        };

        if matched {
            return true;
        }
    }

    false
}

/// Check whether a medication name or class appears in a string list (case-insensitive).
fn medication_in_list(med_name: &str, med_class: &str, list: &[String]) -> bool {
    let name_lower = med_name.to_lowercase();
    let class_lower = med_class.to_lowercase();
    list.iter().any(|item| {
        name_lower.contains(item) || item.contains(&name_lower)
            || class_lower.contains(item) || item.contains(&class_lower)
    })
}

// ──────────────────────────── Tests ──────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Sex;
    

    fn elderly_patient() -> PatientProfile {
        PatientProfile::new(
            78.0, 70.0, Sex::Male,
        )
    }

    fn young_patient() -> PatientProfile {
        PatientProfile::new(
            40.0, 70.0, Sex::Male,
        )
    }

    #[test]
    fn test_defaults_not_empty() {
        let bc = BeersCriteria::with_defaults();
        assert!(bc.len() >= 50, "Expected ≥50 criteria, got {}", bc.len());
    }

    #[test]
    fn test_category_counts() {
        let bc = BeersCriteria::with_defaults();
        let table2 = bc.get_by_category(BeersCategory::AvoidInOlderAdults);
        let table3 = bc.get_by_category(BeersCategory::AvoidDueToCondition);
        let table5 = bc.get_by_category(BeersCategory::AvoidDrugInteraction);
        let table6 = bc.get_by_category(BeersCategory::DoseAdjustmentForRenal);
        assert!(table2.len() >= 10);
        assert!(table3.len() >= 8);
        assert!(table5.len() >= 5);
        assert!(table6.len() >= 10);
    }

    #[test]
    fn test_benzodiazepine_avoid() {
        let bc = BeersCriteria::with_defaults();
        let patient = elderly_patient();
        let med = Medication::new("Diazepam", "benzodiazepine", 5.0);
        let violations = bc.check_single_medication(&med, &patient);
        assert!(!violations.is_empty(), "Diazepam should trigger Beers violation in elderly");
    }

    #[test]
    fn test_no_violation_in_young_patient() {
        let bc = BeersCriteria::with_defaults();
        let patient = young_patient();
        let meds = vec![Medication::new("Diazepam", "benzodiazepine", 5.0)];
        let violations = bc.check_patient(&patient, &meds);
        assert!(violations.is_empty(), "No Beers violations expected for young patients");
    }

    #[test]
    fn test_anticholinergic_with_dementia() {
        let bc = BeersCriteria::with_defaults();
        let patient = elderly_patient().with_conditions(vec![
            Condition::new("G30", "Alzheimer dementia"),
        ]);
        let med = Medication::new("Diphenhydramine", "antihistamine", 25.0);
        let violations = bc.check_single_medication(&med, &patient);
        assert!(!violations.is_empty(), "Diphenhydramine in dementia patient should trigger violation");
    }

    #[test]
    fn test_nsaid_in_ckd() {
        let bc = BeersCriteria::with_defaults();
        let patient = elderly_patient().with_conditions(vec![
            Condition::new("N18.4", "Chronic kidney disease stage 4"),
        ]);
        let med = Medication::new("Ibuprofen", "NSAID", 400.0);
        let violations = bc.check_single_medication(&med, &patient);
        assert!(!violations.is_empty(), "Ibuprofen in CKD patient should trigger violation");
    }

    #[test]
    fn test_renal_dose_adjustment_gabapentin() {
        let bc = BeersCriteria::with_defaults();
        let patient = elderly_patient().with_egfr(25.0);
        let med = Medication::new("Gabapentin", "anticonvulsant", 300.0);
        let meds = vec![med];
        let violations = bc.check_patient(&patient, &meds);
        let renal_violations: Vec<_> = violations
            .iter()
            .filter(|v| v.category == BeersCategory::DoseAdjustmentForRenal)
            .collect();
        assert!(!renal_violations.is_empty(), "Gabapentin should need renal adjustment at eGFR 25");
    }

    #[test]
    fn test_opioid_benzo_interaction() {
        let bc = BeersCriteria::with_defaults();
        let patient = elderly_patient();
        let med_a = Medication::new("Oxycodone", "opioid", 5.0);
        let med_b = Medication::new("Lorazepam", "benzodiazepine", 1.0);
        let violations = bc.check_drug_interaction(&med_a, &med_b, &patient);
        assert!(!violations.is_empty(), "Opioid + benzodiazepine should trigger interaction violation");
    }

    #[test]
    fn test_first_gen_antihistamine_avoid() {
        let bc = BeersCriteria::with_defaults();
        let patient = elderly_patient();
        let med = Medication::new("Hydroxyzine", "antihistamine", 25.0);
        let violations = bc.check_single_medication(&med, &patient);
        assert!(!violations.is_empty(), "Hydroxyzine should be flagged in elderly");
    }

    #[test]
    fn test_tca_avoid() {
        let bc = BeersCriteria::with_defaults();
        let patient = elderly_patient();
        let med = Medication::new("Amitriptyline", "tricyclic antidepressant", 25.0);
        let violations = bc.check_single_medication(&med, &patient);
        assert!(!violations.is_empty(), "Amitriptyline should be flagged in elderly");
    }

    #[test]
    fn test_criterion_composite_weight() {
        let c = BeersCriterion::new(
            "test", BeersCategory::AvoidInOlderAdults,
            &["test_drug"], "test rec", "test rationale",
            QualityOfEvidence::High, StrengthOfRecommendation::Strong,
        );
        assert!((c.composite_weight() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_get_by_id() {
        let bc = BeersCriteria::with_defaults();
        let c = bc.get_by_id("BEERS-2-01");
        assert!(c.is_some());
        assert_eq!(c.unwrap().category, BeersCategory::AvoidInOlderAdults);
    }

    #[test]
    fn test_quality_weight_ordering() {
        assert!(QualityOfEvidence::VeryLow.weight() < QualityOfEvidence::Low.weight());
        assert!(QualityOfEvidence::Low.weight() < QualityOfEvidence::Moderate.weight());
        assert!(QualityOfEvidence::Moderate.weight() < QualityOfEvidence::High.weight());
    }

    #[test]
    fn test_violation_display() {
        let c = BeersCriterion::new(
            "T", BeersCategory::AvoidInOlderAdults,
            &["diazepam"], "avoid", "risk",
            QualityOfEvidence::Moderate, StrengthOfRecommendation::Strong,
        );
        let v = BeersViolation::new(&c, "Diazepam", "test reason");
        let s = format!("{}", v);
        assert!(s.contains("Diazepam"));
    }

    #[test]
    fn test_long_acting_sulfonylurea_avoid() {
        let bc = BeersCriteria::with_defaults();
        let patient = elderly_patient();
        let med = Medication::new("Glyburide", "sulfonylurea", 5.0);
        let violations = bc.check_single_medication(&med, &patient);
        assert!(!violations.is_empty(), "Glyburide should be flagged in elderly");
    }

    #[test]
    fn test_metformin_renal_adjustment() {
        let bc = BeersCriteria::with_defaults();
        let patient = elderly_patient().with_egfr(28.0);
        let med = Medication::new("Metformin", "biguanide", 500.0);
        let meds = vec![med];
        let violations = bc.check_patient(&patient, &meds);
        let renal: Vec<_> = violations.iter()
            .filter(|v| v.category == BeersCategory::DoseAdjustmentForRenal)
            .collect();
        assert!(!renal.is_empty(), "Metformin at eGFR 28 should trigger renal adjustment");
    }

    #[test]
    fn test_check_patient_comprehensive() {
        let bc = BeersCriteria::with_defaults();
        let patient = elderly_patient().with_conditions(vec![
            Condition::new("I50.9", "Heart failure"),
        ]).with_egfr(40.0);
        let meds = vec![
            Medication::new("Ibuprofen", "NSAID", 400.0),
            Medication::new("Gabapentin", "anticonvulsant", 300.0),
        ];
        let violations = bc.check_patient(&patient, &meds);
        assert!(violations.len() >= 2, "Multiple violations expected");
    }
}
