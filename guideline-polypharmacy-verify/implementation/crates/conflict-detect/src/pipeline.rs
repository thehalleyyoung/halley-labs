//! Two-tier verification pipeline for polypharmacy conflict detection.
//!
//! The pipeline first runs a fast **Tier 1** abstract-interpretation screen
//! on every drug pair.  Pairs flagged as potentially interacting are promoted
//! to a slower but more precise **Tier 2** bounded model-checking pass.
//!
//! Both tiers are implemented here as self-contained simulations so the crate
//! compiles without external guardpharma dependencies.

use std::collections::HashMap;
use std::time::Instant;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::types::{
    ConfirmedConflict, ConflictSeverity, ConcentrationInterval, CounterExample,
    DrugId, DrugInfo, InteractionType, MedicationRecord, PatientProfile,
    SafetyCertificate, SafetyVerdict, TraceStep, VerificationResult,
    VerificationTier, PatientId, Dosage, AdministrationRoute, GuidelineId,
};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration knobs for the verification pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Severity threshold for promoting a pair from Tier 1 to Tier 2.
    pub tier1_promotion_threshold: f64,
    /// Maximum number of pairs to process in Tier 2.
    pub max_tier2_pairs: usize,
    /// Simulation time horizon for the PK model (hours).
    pub simulation_horizon_hours: f64,
    /// Time-step resolution for Tier 1 interval analysis (hours).
    pub tier1_time_step_hours: f64,
    /// Time-step resolution for Tier 2 model checking (hours).
    pub tier2_time_step_hours: f64,
    /// Confidence threshold below which a result is marked PossiblySafe.
    pub min_confidence: f64,
    /// Toxic concentration multiplier relative to therapeutic Css.
    pub toxic_multiplier: f64,
    /// Whether to generate full traces for tier 2.
    pub generate_traces: bool,
    /// Whether to generate a safety certificate at the end.
    pub generate_certificate: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            tier1_promotion_threshold: 3.0,
            max_tier2_pairs: 100,
            simulation_horizon_hours: 72.0,
            tier1_time_step_hours: 1.0,
            tier2_time_step_hours: 0.25,
            min_confidence: 0.80,
            toxic_multiplier: 2.5,
            generate_traces: true,
            generate_certificate: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Pipeline result types
// ---------------------------------------------------------------------------

/// Record of a pair being promoted from Tier 1 to Tier 2.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TierTransition {
    pub drug_a: DrugId,
    pub drug_b: DrugId,
    pub tier1_score: f64,
    pub reason: String,
}

/// Aggregate statistics for a pipeline run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineStatistics {
    pub total_pairs_screened: usize,
    pub pairs_promoted_to_tier2: usize,
    pub confirmed_conflicts: usize,
    pub critical_conflicts: usize,
    pub major_conflicts: usize,
    pub moderate_conflicts: usize,
    pub minor_conflicts: usize,
    pub tier1_duration_ms: u64,
    pub tier2_duration_ms: u64,
    pub total_duration_ms: u64,
    pub overall_verdict: SafetyVerdict,
}

/// Complete result of a pipeline run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineResult {
    pub tier1_results: Vec<VerificationResult>,
    pub tier2_results: Vec<VerificationResult>,
    pub confirmed_conflicts: Vec<ConfirmedConflict>,
    pub transitions: Vec<TierTransition>,
    pub certificate: Option<SafetyCertificate>,
    pub statistics: PipelineStatistics,
}

impl PipelineResult {
    /// Overall safety verdict from the pipeline.
    pub fn verdict(&self) -> SafetyVerdict {
        self.statistics.overall_verdict
    }

    /// All results (both tiers) in one iterator.
    pub fn all_results(&self) -> impl Iterator<Item = &VerificationResult> {
        self.tier1_results.iter().chain(self.tier2_results.iter())
    }
}

// ---------------------------------------------------------------------------
// Tier 1 engine – interval-based abstract interpretation
// ---------------------------------------------------------------------------

/// Simulated Tier 1 screening engine using interval arithmetic on PK curves.
struct Tier1Engine<'a> {
    config: &'a PipelineConfig,
}

impl<'a> Tier1Engine<'a> {
    fn new(config: &'a PipelineConfig) -> Self {
        Self { config }
    }

    /// Screen a single drug pair. Returns a score ≥ 0; higher means more
    /// likely to interact.
    fn screen_pair(
        &self,
        med_a: &MedicationRecord,
        med_b: &MedicationRecord,
        patient: &PatientProfile,
    ) -> Tier1ScreenResult {
        let start = Instant::now();

        // Compute baseline PK intervals for each drug
        let interval_a = self.compute_pk_interval(med_a, patient);
        let interval_b = self.compute_pk_interval(med_b, patient);

        // Check CYP interactions
        let cyp_score = self.score_cyp_interaction(&med_a.drug, &med_b.drug);

        // Check protein-binding displacement
        let pb_score = self.score_protein_binding(&med_a.drug, &med_b.drug);

        // Check PD synergy/antagonism (heuristic: same therapeutic class)
        let pd_score = self.score_pd_interaction(&med_a.drug, &med_b.drug);

        // Check QT risk
        let qt_score = self.score_qt_risk(&med_a.drug, &med_b.drug);

        // Interval overlap check: if therapeutic ranges overlap in a dangerous
        // way (both drugs compete for same binding sites), raise score.
        let overlap_score = if interval_a.overlaps(&interval_b) {
            1.5
        } else {
            0.0
        };

        let total_score = cyp_score + pb_score + pd_score + qt_score + overlap_score;

        // Detect interaction types found
        let mut interaction_types = Vec::new();
        if cyp_score > 1.0 {
            let shared = find_shared_cyp(&med_a.drug, &med_b.drug);
            for enzyme in shared {
                interaction_types.push(InteractionType::CypInhibition { enzyme });
            }
        }
        if pb_score > 1.0 {
            interaction_types.push(InteractionType::ProteinBindingDisplacement);
        }
        if qt_score > 1.0 {
            interaction_types.push(InteractionType::QtProlongation);
        }
        if pd_score > 1.0 {
            // Check specific PD interaction types
            if is_serotonergic(&med_a.drug.name, &med_a.drug.therapeutic_class)
                && is_serotonergic(&med_b.drug.name, &med_b.drug.therapeutic_class)
            {
                interaction_types.push(InteractionType::SerotoninSyndrome);
            }
            if is_cns_depressant(&med_a.drug.name, &med_a.drug.therapeutic_class)
                && is_cns_depressant(&med_b.drug.name, &med_b.drug.therapeutic_class)
            {
                interaction_types.push(InteractionType::CnsDepression);
            }
            if is_bleeding_risk(&med_a.drug.name, &med_a.drug.therapeutic_class)
                && is_bleeding_risk(&med_b.drug.name, &med_b.drug.therapeutic_class)
            {
                interaction_types.push(InteractionType::PharmacodynamicSynergy);
            }
            // Fallback: general PD synergy/antagonism for same-class drugs
            if interaction_types.iter().all(|t| !matches!(t,
                InteractionType::SerotoninSyndrome
                | InteractionType::CnsDepression
                | InteractionType::PharmacodynamicSynergy
            )) {
                if med_a.drug.therapeutic_class == med_b.drug.therapeutic_class {
                    interaction_types.push(InteractionType::PharmacodynamicSynergy);
                } else {
                    interaction_types.push(InteractionType::PharmacodynamicAntagonism);
                }
            }
        }

        let elapsed_ms = start.elapsed().as_millis() as u64;

        Tier1ScreenResult {
            drug_a: med_a.drug.id.clone(),
            drug_b: med_b.drug.id.clone(),
            score: total_score,
            interval_a,
            interval_b,
            interaction_types,
            duration_ms: elapsed_ms,
        }
    }

    /// Compute a concentration interval for steady-state conditions.
    fn compute_pk_interval(
        &self,
        med: &MedicationRecord,
        patient: &PatientProfile,
    ) -> ConcentrationInterval {
        let css = med.steady_state_concentration(patient.weight_kg);
        let clearance_adj = patient.combined_clearance_factor();
        let css_adj = if clearance_adj > 0.0 {
            css / clearance_adj
        } else {
            css
        };
        // Interval ± 30% to account for inter-patient variability
        let lo = (css_adj * 0.7).max(0.0);
        let hi = css_adj * 1.3;
        ConcentrationInterval::new(lo, hi)
    }

    fn score_cyp_interaction(&self, a: &DrugInfo, b: &DrugInfo) -> f64 {
        let shared = find_shared_cyp(a, b);
        let count = shared.len() as f64;
        // Each shared CYP enzyme adds risk proportional to the drugs'
        // protein binding (highly bound drugs are more affected).
        count * (1.0 + a.protein_binding + b.protein_binding)
    }

    fn score_protein_binding(&self, a: &DrugInfo, b: &DrugInfo) -> f64 {
        let combined = a.protein_binding + b.protein_binding;
        if combined > 1.5 {
            (combined - 1.0) * 3.0
        } else if combined > 1.0 {
            (combined - 1.0) * 1.5
        } else {
            0.0
        }
    }

    fn score_pd_interaction(&self, a: &DrugInfo, b: &DrugInfo) -> f64 {
        let mut score = 0.0;

        // Same therapeutic class → general PD synergy risk
        if a.therapeutic_class == b.therapeutic_class && a.therapeutic_class != "Unknown" {
            score += 2.5;
        }

        // Serotonin syndrome risk: any two serotonergic drugs
        if is_serotonergic(&a.name, &a.therapeutic_class)
            && is_serotonergic(&b.name, &b.therapeutic_class)
        {
            score += 8.0;
        }

        // CNS depression risk
        if is_cns_depressant(&a.name, &a.therapeutic_class)
            && is_cns_depressant(&b.name, &b.therapeutic_class)
        {
            score += 6.0;
        }

        // Additive bleeding risk (anticoagulant + antiplatelet combos)
        if is_bleeding_risk(&a.name, &a.therapeutic_class)
            && is_bleeding_risk(&b.name, &b.therapeutic_class)
        {
            score += 5.0;
        }

        score
    }

    fn score_qt_risk(&self, a: &DrugInfo, b: &DrugInfo) -> f64 {
        let qt_classes = [
            "Antiarrhythmic",
            "Antibiotic",
            "Antipsychotic",
            "Antidepressant",
        ];
        let a_qt = qt_classes.iter().any(|c| a.therapeutic_class.contains(c));
        let b_qt = qt_classes.iter().any(|c| b.therapeutic_class.contains(c));
        if a_qt && b_qt {
            4.0
        } else if a_qt || b_qt {
            1.0
        } else {
            0.0
        }
    }
}

struct Tier1ScreenResult {
    drug_a: DrugId,
    drug_b: DrugId,
    score: f64,
    interval_a: ConcentrationInterval,
    interval_b: ConcentrationInterval,
    interaction_types: Vec<InteractionType>,
    duration_ms: u64,
}

fn find_shared_cyp(a: &DrugInfo, b: &DrugInfo) -> Vec<String> {
    a.cyp_enzymes
        .iter()
        .filter(|e| b.cyp_enzymes.contains(e))
        .cloned()
        .collect()
}

/// Returns `true` if the drug has clinically significant serotonergic activity.
fn is_serotonergic(name: &str, therapeutic_class: &str) -> bool {
    let lname = name.to_lowercase();
    // SSRIs, SNRIs, TCAs, MAOIs, and specific opioids/other serotonergic agents
    let serotonergic_drugs = [
        "fluoxetine", "sertraline", "paroxetine", "citalopram", "escitalopram",
        "fluvoxamine", "venlafaxine", "duloxetine", "desvenlafaxine",
        "tramadol", "fentanyl", "meperidine", "dextromethorphan",
        "amitriptyline", "nortriptyline", "clomipramine", "imipramine",
        "phenelzine", "tranylcypromine", "selegiline", "linezolid",
        "trazodone", "buspirone", "lithium", "st john",
    ];
    if serotonergic_drugs.iter().any(|d| lname.contains(d)) {
        return true;
    }
    let serotonergic_classes = ["Antidepressant", "SSRI", "SNRI", "MAOI"];
    serotonergic_classes.iter().any(|c| therapeutic_class.contains(c))
}

/// Returns `true` if the drug causes CNS depression.
fn is_cns_depressant(name: &str, therapeutic_class: &str) -> bool {
    let lname = name.to_lowercase();
    let cns_drugs = [
        "morphine", "oxycodone", "hydrocodone", "fentanyl", "methadone",
        "codeine", "tramadol", "diazepam", "lorazepam", "alprazolam",
        "clonazepam", "midazolam", "zolpidem", "phenobarbital",
        "gabapentin", "pregabalin", "baclofen", "carisoprodol",
    ];
    if cns_drugs.iter().any(|d| lname.contains(d)) {
        return true;
    }
    let cns_classes = ["Opioid", "Benzodiazepine", "Barbiturate", "Sedative"];
    cns_classes.iter().any(|c| therapeutic_class.contains(c))
}

/// Returns `true` if the drug contributes to bleeding risk.
fn is_bleeding_risk(name: &str, therapeutic_class: &str) -> bool {
    let lname = name.to_lowercase();
    let bleeding_drugs = [
        "warfarin", "heparin", "enoxaparin", "rivaroxaban", "apixaban",
        "dabigatran", "aspirin", "clopidogrel", "ticagrelor", "prasugrel",
        "ibuprofen", "naproxen", "diclofenac", "ketorolac", "meloxicam",
    ];
    if bleeding_drugs.iter().any(|d| lname.contains(d)) {
        return true;
    }
    let bleeding_classes = [
        "Anticoagulant", "Antiplatelet", "NSAID", "Thrombolytic",
    ];
    bleeding_classes.iter().any(|c| therapeutic_class.contains(c))
}

// ---------------------------------------------------------------------------
// Tier 2 engine – bounded model-checking simulation
// ---------------------------------------------------------------------------

/// Simulated Tier 2 model-checking engine.
struct Tier2Engine<'a> {
    config: &'a PipelineConfig,
}

impl<'a> Tier2Engine<'a> {
    fn new(config: &'a PipelineConfig) -> Self {
        Self { config }
    }

    /// Verify a pair using step-by-step PK simulation with interaction effects.
    fn verify_pair(
        &self,
        med_a: &MedicationRecord,
        med_b: &MedicationRecord,
        patient: &PatientProfile,
        interaction_types: &[InteractionType],
    ) -> Tier2VerifyResult {
        let start = Instant::now();

        let ke_a = med_a.drug.elimination_rate();
        let ke_b = med_b.drug.elimination_rate();
        let vd_a = med_a.drug.estimated_vd_liters(patient.weight_kg);
        let vd_b = med_b.drug.estimated_vd_liters(patient.weight_kg);
        let f_a = med_a.dosage.route.bioavailability_factor() * med_a.drug.bioavailability;
        let f_b = med_b.dosage.route.bioavailability_factor() * med_b.drug.bioavailability;
        let dose_a = med_a.dosage.amount_mg;
        let dose_b = med_b.dosage.amount_mg;
        let tau_a = med_a.dosage.frequency_hours;
        let tau_b = med_b.dosage.frequency_hours;

        // Compute interaction modifier on elimination rate
        let cyp_modifier = self.compute_cyp_modifier(interaction_types);
        let pb_modifier = self.compute_pb_modifier(interaction_types, &med_a.drug, &med_b.drug);

        let dt = self.config.tier2_time_step_hours;
        let n_steps = (self.config.simulation_horizon_hours / dt).ceil() as usize;
        let clearance_factor = patient.combined_clearance_factor();

        let mut conc_a: f64 = 0.0;
        let mut conc_b: f64 = 0.0;
        let mut trace = Vec::with_capacity(n_steps);
        let mut violations: Vec<(f64, String, f64, f64)> = Vec::new();

        // Therapeutic ceilings (simplified: toxic_multiplier × steady-state avg)
        let css_a = med_a.steady_state_concentration(patient.weight_kg);
        let css_b = med_b.steady_state_concentration(patient.weight_kg);
        let toxic_a = css_a * self.config.toxic_multiplier;
        let toxic_b = css_b * self.config.toxic_multiplier;

        for step in 0..n_steps {
            let t = step as f64 * dt;

            // Dose administration events
            if tau_a > 0.0 && (t % tau_a).abs() < dt / 2.0 {
                if vd_a > 0.0 {
                    conc_a += (f_a * dose_a) / vd_a;
                }
            }
            if tau_b > 0.0 && (t % tau_b).abs() < dt / 2.0 {
                if vd_b > 0.0 {
                    conc_b += (f_b * dose_b) / vd_b;
                }
            }

            // Elimination with CYP interaction modifier and clearance
            let effective_ke_a = ke_a * cyp_modifier * clearance_factor;
            let effective_ke_b = ke_b * cyp_modifier * clearance_factor;
            conc_a *= (-effective_ke_a * dt).exp();
            conc_b *= (-effective_ke_b * dt).exp();

            // Protein binding displacement effect: raises free fraction
            conc_a *= 1.0 + pb_modifier * 0.1;
            conc_b *= 1.0 + pb_modifier * 0.1;

            let invariant_holds = conc_a <= toxic_a && conc_b <= toxic_b;

            let mut step_notes = Vec::new();
            if conc_a > toxic_a {
                let msg = format!(
                    "{} exceeds toxic threshold ({:.2} > {:.2})",
                    med_a.drug.name, conc_a, toxic_a
                );
                violations.push((t, msg.clone(), conc_a, toxic_a));
                step_notes.push(msg);
            }
            if conc_b > toxic_b {
                let msg = format!(
                    "{} exceeds toxic threshold ({:.2} > {:.2})",
                    med_b.drug.name, conc_b, toxic_b
                );
                violations.push((t, msg.clone(), conc_b, toxic_b));
                step_notes.push(msg);
            }

            if self.config.generate_traces || !invariant_holds {
                let mut step_record = TraceStep::new(t, if invariant_holds {
                    "within bounds"
                } else {
                    "VIOLATION"
                });
                step_record
                    .drug_concentrations
                    .insert(med_a.drug.id.to_string(), conc_a);
                step_record
                    .drug_concentrations
                    .insert(med_b.drug.id.to_string(), conc_b);
                step_record.safety_invariant_holds = invariant_holds;
                step_record.notes = step_notes;
                trace.push(step_record);
            }
        }

        let elapsed_ms = start.elapsed().as_millis() as u64;

        Tier2VerifyResult {
            drug_a: med_a.drug.id.clone(),
            drug_b: med_b.drug.id.clone(),
            violations,
            trace,
            duration_ms: elapsed_ms,
            toxic_a,
            toxic_b,
        }
    }

    fn compute_cyp_modifier(&self, interactions: &[InteractionType]) -> f64 {
        let mut modifier = 1.0;
        for it in interactions {
            match it {
                InteractionType::CypInhibition { .. } => modifier *= 0.5,
                InteractionType::CypInduction { .. } => modifier *= 1.8,
                _ => {}
            }
        }
        modifier
    }

    fn compute_pb_modifier(
        &self,
        interactions: &[InteractionType],
        a: &DrugInfo,
        b: &DrugInfo,
    ) -> f64 {
        if interactions
            .iter()
            .any(|i| matches!(i, InteractionType::ProteinBindingDisplacement))
        {
            (a.protein_binding + b.protein_binding - 1.0).max(0.0) * 2.0
        } else {
            0.0
        }
    }
}

struct Tier2VerifyResult {
    drug_a: DrugId,
    drug_b: DrugId,
    violations: Vec<(f64, String, f64, f64)>,
    trace: Vec<TraceStep>,
    duration_ms: u64,
    toxic_a: f64,
    toxic_b: f64,
}

// ---------------------------------------------------------------------------
// VerificationPipeline
// ---------------------------------------------------------------------------

/// The main two-tier verification pipeline.
#[derive(Debug, Clone)]
pub struct VerificationPipeline {
    config: PipelineConfig,
}

impl VerificationPipeline {
    pub fn new(config: PipelineConfig) -> Self {
        Self { config }
    }

    pub fn with_defaults() -> Self {
        Self::new(PipelineConfig::default())
    }

    pub fn config(&self) -> &PipelineConfig {
        &self.config
    }

    /// Run the complete pipeline on a patient's medication list.
    pub fn run(&self, patient: &PatientProfile) -> PipelineResult {
        let pipeline_start = Instant::now();
        let meds = &patient.medications;
        let n = meds.len();

        // Generate all unique pairs
        let mut pairs: Vec<(usize, usize)> = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                pairs.push((i, j));
            }
        }

        // ---- Tier 1 screening ----
        let tier1_start = Instant::now();
        let engine1 = Tier1Engine::new(&self.config);
        let mut tier1_results: Vec<VerificationResult> = Vec::new();
        let mut screen_results: Vec<Tier1ScreenResult> = Vec::new();

        for &(i, j) in &pairs {
            let screen = engine1.screen_pair(&meds[i], &meds[j], patient);
            let verdict = if screen.score < self.config.tier1_promotion_threshold {
                SafetyVerdict::Safe
            } else if screen.score < self.config.tier1_promotion_threshold * 2.0 {
                SafetyVerdict::PossiblySafe
            } else {
                SafetyVerdict::PossiblyUnsafe
            };

            tier1_results.push(VerificationResult {
                drug_pair: (screen.drug_a.clone(), screen.drug_b.clone()),
                tier: VerificationTier::Tier1Abstract,
                verdict,
                conflicts: Vec::new(),
                trace: None,
                duration_ms: screen.duration_ms,
                notes: vec![format!("Tier 1 score: {:.2}", screen.score)],
            });
            screen_results.push(screen);
        }
        let tier1_ms = tier1_start.elapsed().as_millis() as u64;

        // ---- Decide promotions ----
        let mut promotions: Vec<(usize, TierTransition)> = Vec::new();
        for (idx, sr) in screen_results.iter().enumerate() {
            if sr.score >= self.config.tier1_promotion_threshold {
                let reason = if sr.score >= self.config.tier1_promotion_threshold * 2.0 {
                    format!(
                        "High interaction score {:.2} (>{:.2})",
                        sr.score,
                        self.config.tier1_promotion_threshold * 2.0
                    )
                } else {
                    format!(
                        "Moderate interaction score {:.2} (>{:.2})",
                        sr.score, self.config.tier1_promotion_threshold
                    )
                };
                promotions.push((
                    idx,
                    TierTransition {
                        drug_a: sr.drug_a.clone(),
                        drug_b: sr.drug_b.clone(),
                        tier1_score: sr.score,
                        reason,
                    },
                ));
            }
        }

        // Limit to max_tier2_pairs, sorted by score descending
        promotions.sort_by(|a, b| {
            b.1.tier1_score
                .partial_cmp(&a.1.tier1_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        promotions.truncate(self.config.max_tier2_pairs);

        let transitions: Vec<TierTransition> =
            promotions.iter().map(|(_, t)| t.clone()).collect();

        // ---- Tier 2 verification ----
        let tier2_start = Instant::now();
        let engine2 = Tier2Engine::new(&self.config);
        let mut tier2_results: Vec<VerificationResult> = Vec::new();
        let mut all_conflicts: Vec<ConfirmedConflict> = Vec::new();
        let mut conflict_counter = 0u32;

        for (pair_idx, _transition) in &promotions {
            let (i, j) = pairs[*pair_idx];
            let sr = &screen_results[*pair_idx];

            let t2 = engine2.verify_pair(&meds[i], &meds[j], patient, &sr.interaction_types);

            let mut pair_conflicts = Vec::new();
            if !t2.violations.is_empty() {
                // Build counterexample from the first violation
                let (vtime, vdesc, _vconc, _vtoxic) = &t2.violations[0];
                let counter_example = CounterExample {
                    trace: t2
                        .trace
                        .iter()
                        .filter(|s| {
                            s.time_hours <= *vtime + self.config.tier2_time_step_hours
                        })
                        .cloned()
                        .collect(),
                    violated_property: format!(
                        "C_max < toxic threshold for all drugs at all times"
                    ),
                    violation_time_hours: *vtime,
                    description: vdesc.clone(),
                };

                // Determine severity based on how much the threshold is exceeded
                let max_ratio = t2.violations.iter().fold(0.0_f64, |acc, (_, _, conc, tox)| {
                    if *tox > 0.0 {
                        acc.max(conc / tox)
                    } else {
                        acc
                    }
                });
                let severity = if max_ratio > 3.0 {
                    ConflictSeverity::Critical
                } else if max_ratio > 2.0 {
                    ConflictSeverity::Major
                } else if max_ratio > 1.5 {
                    ConflictSeverity::Moderate
                } else {
                    ConflictSeverity::Minor
                };

                let verdict = match severity {
                    ConflictSeverity::Critical => SafetyVerdict::Unsafe,
                    ConflictSeverity::Major => SafetyVerdict::PossiblyUnsafe,
                    _ => SafetyVerdict::PossiblySafe,
                };

                for itype in &sr.interaction_types {
                    // Only create PK-severity conflicts for PK-category interactions.
                    // PD interactions get their own severity in the PD block below.
                    if matches!(itype,
                        InteractionType::SerotoninSyndrome
                        | InteractionType::CnsDepression
                        | InteractionType::QtProlongation
                    ) {
                        continue;
                    }
                    conflict_counter += 1;
                    let drugs = vec![t2.drug_a.clone(), t2.drug_b.clone()];
                    let cid = ConfirmedConflict::generate_id(&drugs, itype);
                    let confidence = (1.0 - 1.0 / (t2.violations.len() as f64 + 1.0)).min(0.99);

                    let conflict = ConfirmedConflict {
                        id: cid,
                        drugs: drugs.clone(),
                        interaction_type: itype.clone(),
                        severity,
                        verdict,
                        mechanism_description: itype.description(),
                        evidence_tier: VerificationTier::Tier2ModelCheck,
                        counter_example: Some(counter_example.clone()),
                        confidence,
                        clinical_recommendation: generate_recommendation(severity, itype),
                        affected_parameters: affected_params_for(itype),
                        guideline_references: Vec::new(),
                    };
                    pair_conflicts.push(conflict);
                }
            }

            // PD-aware conflict generation: pharmacodynamic interactions
            // (serotonin syndrome, CNS depression, etc.) are dangerous
            // regardless of whether PK concentration thresholds are breached.
            {
                let pd_types: Vec<_> = sr.interaction_types.iter().filter(|t| {
                    matches!(t,
                        InteractionType::SerotoninSyndrome
                        | InteractionType::CnsDepression
                        | InteractionType::QtProlongation
                    )
                }).collect();

                for itype in &pd_types {
                    // Skip if we already have a conflict for this exact interaction type
                    let already_present = pair_conflicts.iter().any(|c| &c.interaction_type == *itype);
                    if already_present {
                        continue;
                    }
                    conflict_counter += 1;
                    let drugs = vec![t2.drug_a.clone(), t2.drug_b.clone()];
                    let cid = ConfirmedConflict::generate_id(&drugs, itype);
                    let severity = match itype {
                        InteractionType::SerotoninSyndrome => ConflictSeverity::Critical,
                        InteractionType::QtProlongation => ConflictSeverity::Critical,
                        InteractionType::CnsDepression => ConflictSeverity::Major,
                        _ => ConflictSeverity::Moderate,
                    };
                    let verdict = match severity {
                        ConflictSeverity::Critical => SafetyVerdict::Unsafe,
                        ConflictSeverity::Major => SafetyVerdict::PossiblyUnsafe,
                        _ => SafetyVerdict::PossiblySafe,
                    };
                    let conflict = ConfirmedConflict {
                        id: cid,
                        drugs: drugs.clone(),
                        interaction_type: (*itype).clone(),
                        severity,
                        verdict,
                        mechanism_description: itype.description(),
                        evidence_tier: VerificationTier::Tier2ModelCheck,
                        counter_example: None,
                        confidence: 0.90,
                        clinical_recommendation: generate_recommendation(severity, itype),
                        affected_parameters: affected_params_for(itype),
                        guideline_references: Vec::new(),
                    };
                    pair_conflicts.push(conflict);
                }
            }

            let pair_verdict = if pair_conflicts.is_empty() {
                SafetyVerdict::Safe
            } else {
                pair_conflicts
                    .iter()
                    .map(|c| c.verdict)
                    .fold(SafetyVerdict::Safe, SafetyVerdict::merge)
            };

            all_conflicts.extend(pair_conflicts.clone());

            tier2_results.push(VerificationResult {
                drug_pair: (t2.drug_a, t2.drug_b),
                tier: VerificationTier::Tier2ModelCheck,
                verdict: pair_verdict,
                conflicts: pair_conflicts,
                trace: if self.config.generate_traces {
                    Some(t2.trace)
                } else {
                    None
                },
                duration_ms: t2.duration_ms,
                notes: vec![format!(
                    "Tier 2: {} violations in {:.1}h simulation",
                    t2.violations.len(),
                    self.config.simulation_horizon_hours
                )],
            });
        }
        let tier2_ms = tier2_start.elapsed().as_millis() as u64;
        let total_ms = pipeline_start.elapsed().as_millis() as u64;

        // ---- Statistics ----
        let mut critical = 0usize;
        let mut major = 0usize;
        let mut moderate = 0usize;
        let mut minor = 0usize;
        for c in &all_conflicts {
            match c.severity {
                ConflictSeverity::Critical => critical += 1,
                ConflictSeverity::Major => major += 1,
                ConflictSeverity::Moderate => moderate += 1,
                ConflictSeverity::Minor => minor += 1,
            }
        }

        let overall_verdict = if critical > 0 {
            SafetyVerdict::Unsafe
        } else if major > 0 {
            SafetyVerdict::PossiblyUnsafe
        } else if moderate > 0 || minor > 0 {
            SafetyVerdict::PossiblySafe
        } else {
            SafetyVerdict::Safe
        };

        let stats = PipelineStatistics {
            total_pairs_screened: pairs.len(),
            pairs_promoted_to_tier2: promotions.len(),
            confirmed_conflicts: all_conflicts.len(),
            critical_conflicts: critical,
            major_conflicts: major,
            moderate_conflicts: moderate,
            minor_conflicts: minor,
            tier1_duration_ms: tier1_ms,
            tier2_duration_ms: tier2_ms,
            total_duration_ms: total_ms,
            overall_verdict,
        };

        // ---- Certificate ----
        let certificate = if self.config.generate_certificate {
            Some(SafetyCertificate {
                id: format!("cert-{}", patient.id),
                patient_id: patient.id.clone(),
                medications: patient.drug_ids(),
                verdict: overall_verdict,
                conflicts: all_conflicts.clone(),
                methodology: "Two-tier verification: Tier 1 interval-based abstract \
                    interpretation + Tier 2 bounded model checking"
                    .to_string(),
                assumptions: vec![
                    "One-compartment PK model with first-order elimination".to_string(),
                    "Steady-state conditions reached".to_string(),
                    format!(
                        "Simulation horizon: {} hours",
                        self.config.simulation_horizon_hours
                    ),
                    "Inter-patient variability modelled as ±30% on Css".to_string(),
                ],
                evidence_summary: tier2_results
                    .iter()
                    .map(|r| {
                        format!(
                            "{} vs {}: {} ({} conflicts)",
                            r.drug_pair.0,
                            r.drug_pair.1,
                            r.verdict,
                            r.conflicts.len()
                        )
                    })
                    .collect(),
                generated_at: "2025-01-01T00:00:00Z".to_string(),
                valid_until: Some("2025-04-01T00:00:00Z".to_string()),
                confidence_score: if all_conflicts.is_empty() {
                    0.95
                } else {
                    all_conflicts
                        .iter()
                        .map(|c| c.confidence)
                        .sum::<f64>()
                        / all_conflicts.len() as f64
                },
            })
        } else {
            None
        };

        PipelineResult {
            tier1_results,
            tier2_results,
            confirmed_conflicts: all_conflicts,
            transitions,
            certificate,
            statistics: stats,
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn generate_recommendation(severity: ConflictSeverity, itype: &InteractionType) -> String {
    let action = match severity {
        ConflictSeverity::Critical => "Contraindicated: avoid concomitant use",
        ConflictSeverity::Major => "Consider alternative therapy or dose adjustment",
        ConflictSeverity::Moderate => "Monitor closely and adjust dose if needed",
        ConflictSeverity::Minor => "Be aware; usually no intervention needed",
    };
    format!("{}. Mechanism: {}", action, itype.description())
}

fn affected_params_for(itype: &InteractionType) -> Vec<String> {
    match itype {
        InteractionType::CypInhibition { enzyme } => {
            vec![
                format!("CYP {} activity", enzyme),
                "plasma concentration".to_string(),
                "AUC".to_string(),
            ]
        }
        InteractionType::CypInduction { enzyme } => {
            vec![
                format!("CYP {} activity", enzyme),
                "plasma concentration".to_string(),
                "clearance".to_string(),
            ]
        }
        InteractionType::ProteinBindingDisplacement => {
            vec![
                "free drug fraction".to_string(),
                "volume of distribution".to_string(),
            ]
        }
        InteractionType::RenalCompetition => {
            vec![
                "renal clearance".to_string(),
                "half-life".to_string(),
            ]
        }
        InteractionType::PharmacodynamicSynergy => {
            vec!["pharmacodynamic effect".to_string(), "toxicity risk".to_string()]
        }
        InteractionType::PharmacodynamicAntagonism => {
            vec!["therapeutic efficacy".to_string()]
        }
        InteractionType::AbsorptionAlteration => {
            vec!["bioavailability".to_string(), "Tmax".to_string()]
        }
        InteractionType::QtProlongation => {
            vec![
                "QTc interval".to_string(),
                "cardiac rhythm".to_string(),
                "torsades risk".to_string(),
            ]
        }
        InteractionType::SerotoninSyndrome => {
            vec![
                "serotonergic tone".to_string(),
                "neuromuscular excitation".to_string(),
                "autonomic instability".to_string(),
            ]
        }
        InteractionType::CnsDepression => {
            vec![
                "respiratory drive".to_string(),
                "sedation level".to_string(),
                "consciousness".to_string(),
            ]
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::OrganFunction;

    fn make_drug(id: &str, name: &str, class: &str, cyps: &[&str], half_life: f64) -> DrugInfo {
        DrugInfo {
            id: DrugId::new(id),
            name: name.to_string(),
            therapeutic_class: class.to_string(),
            cyp_enzymes: cyps.iter().map(|s| s.to_string()).collect(),
            half_life_hours: half_life,
            bioavailability: 0.8,
            protein_binding: 0.6,
            therapeutic_index: None,
        }
    }

    fn make_med(drug: DrugInfo, dose_mg: f64, freq_h: f64) -> MedicationRecord {
        MedicationRecord::new(drug, Dosage::new(dose_mg, freq_h, AdministrationRoute::Oral))
    }

    fn make_patient(meds: Vec<MedicationRecord>) -> PatientProfile {
        PatientProfile {
            id: PatientId::new("test-patient"),
            age: 65,
            weight_kg: 70.0,
            medications: meds,
            conditions: vec![],
            allergies: vec![],
            renal_function: OrganFunction::Normal,
            hepatic_function: OrganFunction::Normal,
        }
    }

    #[test]
    fn test_pipeline_no_medications() {
        let pipe = VerificationPipeline::with_defaults();
        let patient = make_patient(vec![]);
        let result = pipe.run(&patient);
        assert_eq!(result.statistics.total_pairs_screened, 0);
        assert!(result.confirmed_conflicts.is_empty());
        assert_eq!(result.verdict(), SafetyVerdict::Safe);
    }

    #[test]
    fn test_pipeline_single_medication() {
        let drug = make_drug("aspirin", "Aspirin", "NSAID", &["2C9"], 4.0);
        let med = make_med(drug, 325.0, 6.0);
        let pipe = VerificationPipeline::with_defaults();
        let patient = make_patient(vec![med]);
        let result = pipe.run(&patient);
        assert_eq!(result.statistics.total_pairs_screened, 0);
        assert!(result.confirmed_conflicts.is_empty());
    }

    #[test]
    fn test_pipeline_non_interacting_pair() {
        let drug_a = make_drug("aspirin", "Aspirin", "NSAID", &["2C9"], 4.0);
        let drug_b = make_drug("metformin", "Metformin", "Antidiabetic", &[], 6.0);
        let med_a = make_med(drug_a, 325.0, 6.0);
        let med_b = make_med(drug_b, 500.0, 8.0);
        let pipe = VerificationPipeline::with_defaults();
        let patient = make_patient(vec![med_a, med_b]);
        let result = pipe.run(&patient);
        assert_eq!(result.statistics.total_pairs_screened, 1);
    }

    #[test]
    fn test_pipeline_cyp_interacting_pair() {
        let drug_a = make_drug("warfarin", "Warfarin", "Anticoagulant", &["2C9", "3A4"], 40.0);
        let drug_b = make_drug("fluconazole", "Fluconazole", "Antifungal", &["2C9", "3A4"], 30.0);
        let med_a = make_med(drug_a, 5.0, 24.0);
        let med_b = make_med(drug_b, 200.0, 24.0);

        let config = PipelineConfig {
            tier1_promotion_threshold: 2.0,
            ..Default::default()
        };
        let pipe = VerificationPipeline::new(config);
        let patient = make_patient(vec![med_a, med_b]);
        let result = pipe.run(&patient);

        assert!(
            result.statistics.pairs_promoted_to_tier2 >= 1,
            "CYP-interacting pair should be promoted"
        );
        assert!(!result.transitions.is_empty());
    }

    #[test]
    fn test_pipeline_same_class_pd_interaction() {
        let drug_a = make_drug("amiodarone", "Amiodarone", "Antiarrhythmic", &["3A4"], 50.0);
        let drug_b = make_drug("sotalol", "Sotalol", "Antiarrhythmic", &["2D6"], 12.0);
        let med_a = make_med(drug_a, 200.0, 24.0);
        let med_b = make_med(drug_b, 80.0, 12.0);

        let config = PipelineConfig {
            tier1_promotion_threshold: 2.0,
            ..Default::default()
        };
        let pipe = VerificationPipeline::new(config);
        let patient = make_patient(vec![med_a, med_b]);
        let result = pipe.run(&patient);

        assert!(result.statistics.pairs_promoted_to_tier2 >= 1);
    }

    #[test]
    fn test_pipeline_certificate_generated() {
        let drug_a = make_drug("drug_a", "Drug A", "ClassX", &[], 10.0);
        let drug_b = make_drug("drug_b", "Drug B", "ClassY", &[], 8.0);
        let med_a = make_med(drug_a, 100.0, 12.0);
        let med_b = make_med(drug_b, 50.0, 8.0);

        let pipe = VerificationPipeline::with_defaults();
        let patient = make_patient(vec![med_a, med_b]);
        let result = pipe.run(&patient);

        assert!(result.certificate.is_some());
        let cert = result.certificate.unwrap();
        assert_eq!(cert.patient_id, PatientId::new("test-patient"));
    }

    #[test]
    fn test_pipeline_statistics_counts() {
        let drugs: Vec<DrugInfo> = (0..4)
            .map(|i| make_drug(&format!("d{}", i), &format!("Drug{}", i), "General", &[], 10.0))
            .collect();
        let meds: Vec<MedicationRecord> = drugs
            .into_iter()
            .map(|d| make_med(d, 100.0, 12.0))
            .collect();

        let pipe = VerificationPipeline::with_defaults();
        let patient = make_patient(meds);
        let result = pipe.run(&patient);

        // 4 choose 2 = 6 pairs
        assert_eq!(result.statistics.total_pairs_screened, 6);
        assert_eq!(result.tier1_results.len(), 6);
    }

    #[test]
    fn test_pipeline_no_certificate_when_disabled() {
        let config = PipelineConfig {
            generate_certificate: false,
            ..Default::default()
        };
        let pipe = VerificationPipeline::new(config);
        let patient = make_patient(vec![]);
        let result = pipe.run(&patient);
        assert!(result.certificate.is_none());
    }

    #[test]
    fn test_tier_transition_fields() {
        let tt = TierTransition {
            drug_a: DrugId::new("a"),
            drug_b: DrugId::new("b"),
            tier1_score: 5.5,
            reason: "High score".to_string(),
        };
        assert!(tt.tier1_score > 5.0);
        assert!(tt.reason.contains("High"));
    }

    #[test]
    fn test_pipeline_result_all_results() {
        let drug_a = make_drug("x", "X", "C", &["3A4"], 12.0);
        let drug_b = make_drug("y", "Y", "C", &["3A4"], 12.0);
        let med_a = make_med(drug_a, 100.0, 12.0);
        let med_b = make_med(drug_b, 100.0, 12.0);

        let config = PipelineConfig {
            tier1_promotion_threshold: 1.0,
            ..Default::default()
        };
        let pipe = VerificationPipeline::new(config);
        let patient = make_patient(vec![med_a, med_b]);
        let result = pipe.run(&patient);

        let total: usize = result.all_results().count();
        assert!(total >= 1);
    }

    #[test]
    fn test_pipeline_impaired_organs() {
        let drug = make_drug("digoxin", "Digoxin", "Cardiac Glycoside", &[], 36.0);
        let med = make_med(drug.clone(), 0.25, 24.0);
        let drug2 = make_drug("amiodarone", "Amiodarone", "Antiarrhythmic", &["3A4"], 50.0);
        let med2 = make_med(drug2, 200.0, 24.0);

        let mut patient = make_patient(vec![med, med2]);
        patient.renal_function = OrganFunction::ModerateImpairment;
        patient.hepatic_function = OrganFunction::MildImpairment;

        let pipe = VerificationPipeline::with_defaults();
        let result = pipe.run(&patient);

        assert_eq!(result.statistics.total_pairs_screened, 1);
    }

    #[test]
    fn test_generate_recommendation_content() {
        let rec = generate_recommendation(
            ConflictSeverity::Critical,
            &InteractionType::QtProlongation,
        );
        assert!(rec.contains("Contraindicated"));
        assert!(rec.contains("QT"));
    }

    #[test]
    fn test_affected_params_cyp() {
        let params = affected_params_for(&InteractionType::CypInhibition {
            enzyme: "3A4".to_string(),
        });
        assert!(params.iter().any(|p| p.contains("3A4")));
        assert!(params.iter().any(|p| p.contains("plasma")));
    }
}
