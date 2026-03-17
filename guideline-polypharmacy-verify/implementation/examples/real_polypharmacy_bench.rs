//! Real polypharmacy verification benchmark.
//!
//! Exercises the GuardPharma two-tier verification pipeline (Tier 1 abstract
//! interpretation + Tier 2 bounded model checking) on 10 clinically motivated
//! drug combination scenarios.  Each scenario has a ground-truth label
//! (DANGEROUS / SAFE) based on published drug-interaction references.
//!
//! BASELINE: A simple pairwise CYP-overlap lookup table that flags any pair
//! sharing a CYP enzyme as "interacting".  This is roughly what a naïve
//! drug-interaction checker does.
//!
//! Run with: `cargo run --example real_polypharmacy_bench --release`

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use guardpharma_conflict_detect::{
    AdministrationRoute, Dosage, DrugId, DrugInfo, MedicationRecord,
    OrganFunction, PatientId, PatientProfile, SafetyVerdict,
    VerificationPipeline,
};

// ═══════════════════════════════════════════════════════════════════════════
// Drug catalog – realistic PK parameters from published sources
// ═══════════════════════════════════════════════════════════════════════════

fn drug(
    id: &str,
    name: &str,
    class: &str,
    cyps: &[&str],
    half_life: f64,
    bioavail: f64,
    protein_bind: f64,
    ti: Option<f64>,
) -> DrugInfo {
    DrugInfo {
        id: DrugId::new(id),
        name: name.to_string(),
        therapeutic_class: class.to_string(),
        cyp_enzymes: cyps.iter().map(|s| s.to_string()).collect(),
        half_life_hours: half_life,
        bioavailability: bioavail,
        protein_binding: protein_bind,
        therapeutic_index: ti,
    }
}

fn med(info: DrugInfo, dose_mg: f64, freq_h: f64) -> MedicationRecord {
    MedicationRecord::new(
        info,
        Dosage::new(dose_mg, freq_h, AdministrationRoute::Oral),
    )
}

// Catalog of drugs with published PK values.
// Sources: Goodman & Gilman 13th ed., Lexicomp, FDA labels.
fn warfarin() -> DrugInfo {
    drug("warfarin", "Warfarin", "Anticoagulant",
         &["CYP2C9", "CYP3A4", "CYP1A2"], 40.0, 0.95, 0.99, Some(2.0))
}
fn fluconazole() -> DrugInfo {
    drug("fluconazole", "Fluconazole", "Antifungal",
         &["CYP2C9", "CYP3A4", "CYP2C19"], 30.0, 0.90, 0.12, None)
}
fn amiodarone() -> DrugInfo {
    drug("amiodarone", "Amiodarone", "Antiarrhythmic",
         &["CYP2C9", "CYP3A4", "CYP2D6"], 58.0, 0.50, 0.96, Some(2.5))
}
fn metformin() -> DrugInfo {
    drug("metformin", "Metformin", "Antidiabetic",
         &[], 5.0, 0.55, 0.01, None)
}
fn lisinopril() -> DrugInfo {
    drug("lisinopril", "Lisinopril", "ACE Inhibitor",
         &[], 12.0, 0.25, 0.0, None)
}
fn amlodipine() -> DrugInfo {
    drug("amlodipine", "Amlodipine", "Calcium Channel Blocker",
         &["CYP3A4"], 40.0, 0.64, 0.97, None)
}
fn atorvastatin() -> DrugInfo {
    drug("atorvastatin", "Atorvastatin", "Statin",
         &["CYP3A4"], 14.0, 0.12, 0.98, None)
}
fn omeprazole() -> DrugInfo {
    drug("omeprazole", "Omeprazole", "PPI",
         &["CYP2C19", "CYP3A4"], 1.0, 0.40, 0.95, None)
}
fn clopidogrel() -> DrugInfo {
    drug("clopidogrel", "Clopidogrel", "Antiplatelet",
         &["CYP2C19", "CYP3A4", "CYP2B6"], 6.0, 0.50, 0.98, None)
}
fn simvastatin() -> DrugInfo {
    drug("simvastatin", "Simvastatin", "Statin",
         &["CYP3A4"], 3.0, 0.05, 0.95, None)
}
fn clarithromycin() -> DrugInfo {
    drug("clarithromycin", "Clarithromycin", "Antibiotic",
         &["CYP3A4"], 5.0, 0.50, 0.70, None)
}
fn metoprolol() -> DrugInfo {
    drug("metoprolol", "Metoprolol", "Beta Blocker",
         &["CYP2D6"], 4.0, 0.50, 0.12, None)
}
fn fluoxetine() -> DrugInfo {
    drug("fluoxetine", "Fluoxetine", "Antidepressant",
         &["CYP2D6", "CYP2C9", "CYP3A4"], 72.0, 0.72, 0.95, None)
}
fn aspirin() -> DrugInfo {
    drug("aspirin", "Aspirin", "Antiplatelet",
         &[], 0.3, 0.68, 0.80, None)
}
fn levothyroxine() -> DrugInfo {
    drug("levothyroxine", "Levothyroxine", "Thyroid",
         &[], 168.0, 0.80, 0.99, Some(2.0))
}
fn gabapentin() -> DrugInfo {
    drug("gabapentin", "Gabapentin", "Anticonvulsant",
         &[], 6.0, 0.60, 0.0, None)
}
fn _digoxin() -> DrugInfo {
    drug("digoxin", "Digoxin", "Cardiac Glycoside",
         &[], 40.0, 0.75, 0.25, Some(1.5))
}
fn sertraline() -> DrugInfo {
    drug("sertraline", "Sertraline", "Antidepressant",
         &["CYP2C19", "CYP2D6", "CYP3A4"], 26.0, 0.44, 0.98, None)
}
fn tramadol() -> DrugInfo {
    drug("tramadol", "Tramadol", "Opioid",
         &["CYP2D6", "CYP3A4"], 6.0, 0.75, 0.20, None)
}
fn _carbamazepine() -> DrugInfo {
    drug("carbamazepine", "Carbamazepine", "Anticonvulsant",
         &["CYP3A4", "CYP2C8"], 18.0, 0.75, 0.76, Some(2.5))
}

// ═══════════════════════════════════════════════════════════════════════════
// Test scenarios with ground-truth labels
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GroundTruth {
    Dangerous,
    Safe,
}

struct Scenario {
    name: &'static str,
    description: &'static str,
    meds: Vec<MedicationRecord>,
    ground_truth: GroundTruth,
    expected_mechanism: &'static str,
}

fn build_patient(meds: Vec<MedicationRecord>) -> PatientProfile {
    PatientProfile {
        id: PatientId::new("bench-patient"),
        age: 72,
        weight_kg: 78.0,
        medications: meds,
        conditions: vec![],
        allergies: vec![],
        renal_function: OrganFunction::Normal,
        hepatic_function: OrganFunction::Normal,
    }
}

fn scenarios() -> Vec<Scenario> {
    vec![
        // ── DANGEROUS combinations (known interactions) ──────────────
        Scenario {
            name: "Warfarin + Fluconazole + Amiodarone",
            description: "Triple CYP2C9/3A4 inhibition cascade → hemorrhage risk",
            meds: vec![
                med(warfarin(), 5.0, 24.0),
                med(fluconazole(), 200.0, 24.0),
                med(amiodarone(), 200.0, 24.0),
            ],
            ground_truth: GroundTruth::Dangerous,
            expected_mechanism: "CYP2C9 + CYP3A4 inhibition",
        },
        Scenario {
            name: "Simvastatin + Clarithromycin",
            description: "CYP3A4 inhibition → rhabdomyolysis risk (FDA boxed warning)",
            meds: vec![
                med(simvastatin(), 40.0, 24.0),
                med(clarithromycin(), 500.0, 12.0),
            ],
            ground_truth: GroundTruth::Dangerous,
            expected_mechanism: "CYP3A4 inhibition",
        },
        Scenario {
            name: "Clopidogrel + Omeprazole",
            description: "CYP2C19 inhibition reduces clopidogrel activation (FDA warning)",
            meds: vec![
                med(clopidogrel(), 75.0, 24.0),
                med(omeprazole(), 20.0, 24.0),
            ],
            ground_truth: GroundTruth::Dangerous,
            expected_mechanism: "CYP2C19 inhibition",
        },
        Scenario {
            name: "Fluoxetine + Metoprolol + Tramadol",
            description: "Triple CYP2D6 competition → bradycardia + serotonin syndrome risk",
            meds: vec![
                med(fluoxetine(), 20.0, 24.0),
                med(metoprolol(), 50.0, 12.0),
                med(tramadol(), 50.0, 6.0),
            ],
            ground_truth: GroundTruth::Dangerous,
            expected_mechanism: "CYP2D6 inhibition",
        },
        Scenario {
            name: "Warfarin + Aspirin + Clopidogrel",
            description: "Triple antithrombotic → major bleeding risk",
            meds: vec![
                med(warfarin(), 5.0, 24.0),
                med(aspirin(), 81.0, 24.0),
                med(clopidogrel(), 75.0, 24.0),
            ],
            ground_truth: GroundTruth::Dangerous,
            expected_mechanism: "Pharmacodynamic synergy (antithrombotic)",
        },
        Scenario {
            name: "Sertraline + Tramadol",
            description: "SSRI + opioid → serotonin syndrome risk via CYP2D6 + PD",
            meds: vec![
                med(sertraline(), 100.0, 24.0),
                med(tramadol(), 50.0, 6.0),
            ],
            ground_truth: GroundTruth::Dangerous,
            expected_mechanism: "CYP2D6 + pharmacodynamic synergy",
        },

        // ── SAFE combinations (no known significant interactions) ────
        Scenario {
            name: "Metformin + Lisinopril + Amlodipine",
            description: "Standard diabetes + hypertension regimen, no CYP overlap",
            meds: vec![
                med(metformin(), 1000.0, 12.0),
                med(lisinopril(), 20.0, 24.0),
                med(amlodipine(), 5.0, 24.0),
            ],
            ground_truth: GroundTruth::Safe,
            expected_mechanism: "None (no shared CYP)",
        },
        Scenario {
            name: "Levothyroxine + Gabapentin",
            description: "Thyroid + neuropathic pain, orthogonal metabolism",
            meds: vec![
                med(levothyroxine(), 0.1, 24.0),
                med(gabapentin(), 300.0, 8.0),
            ],
            ground_truth: GroundTruth::Safe,
            expected_mechanism: "None (no CYP involvement)",
        },
        Scenario {
            name: "Metformin + Lisinopril + Atorvastatin + Aspirin",
            description: "Standard cardiac prevention quartet, well-studied safe combo",
            meds: vec![
                med(metformin(), 1000.0, 12.0),
                med(lisinopril(), 20.0, 24.0),
                med(atorvastatin(), 40.0, 24.0),
                med(aspirin(), 81.0, 24.0),
            ],
            ground_truth: GroundTruth::Safe,
            expected_mechanism: "None (minimal CYP overlap)",
        },
        Scenario {
            name: "Amlodipine + Metoprolol + Lisinopril + Metformin + Gabapentin",
            description: "5-drug regimen for HTN+DM+neuropathy, no major interactions",
            meds: vec![
                med(amlodipine(), 5.0, 24.0),
                med(metoprolol(), 50.0, 12.0),
                med(lisinopril(), 20.0, 24.0),
                med(metformin(), 1000.0, 12.0),
                med(gabapentin(), 300.0, 8.0),
            ],
            ground_truth: GroundTruth::Safe,
            expected_mechanism: "None (diverse metabolism)",
        },
    ]
}

// ═══════════════════════════════════════════════════════════════════════════
// Baseline: simple pairwise CYP-overlap lookup table
// ═══════════════════════════════════════════════════════════════════════════

/// Baseline checker: flags any pair that shares a CYP enzyme.
/// This is the simplest possible "interaction detector" — no PK modeling.
struct BaselineLookup {
    /// Map from drug name → set of CYP enzymes
    cyp_map: HashMap<String, HashSet<String>>,
}

impl BaselineLookup {
    fn new(meds: &[MedicationRecord]) -> Self {
        let mut cyp_map = HashMap::new();
        for m in meds {
            let enzymes: HashSet<String> = m.drug.cyp_enzymes.iter().cloned().collect();
            cyp_map.insert(m.drug.id.as_str().to_string(), enzymes);
        }
        BaselineLookup { cyp_map }
    }

    /// Returns (n_flagged_pairs, total_pairs, verdict_is_dangerous)
    fn check(&self) -> (usize, usize, bool) {
        let drugs: Vec<&String> = self.cyp_map.keys().collect();
        let n = drugs.len();
        let mut flagged = 0;
        let mut total = 0;

        for i in 0..n {
            for j in (i + 1)..n {
                total += 1;
                let a = &self.cyp_map[drugs[i]];
                let b = &self.cyp_map[drugs[j]];
                if !a.is_empty() && !b.is_empty() && !a.is_disjoint(b) {
                    flagged += 1;
                }
            }
        }

        (flagged, total, flagged > 0)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Benchmark runner
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug)]
struct ScenarioResult {
    name: String,
    n_drugs: usize,
    ground_truth: GroundTruth,
    // GuardPharma
    gp_verdict: SafetyVerdict,
    gp_conflicts: usize,
    gp_promoted: usize,
    gp_tier1_ms: u64,
    gp_tier2_ms: u64,
    gp_total_ms: u64,
    gp_correct: bool,
    // Baseline
    bl_flagged: usize,
    bl_total_pairs: usize,
    bl_verdict_dangerous: bool,
    bl_correct: bool,
    bl_time_us: u64,
}

fn classify_verdict(v: SafetyVerdict) -> GroundTruth {
    match v {
        SafetyVerdict::Safe | SafetyVerdict::PossiblySafe => GroundTruth::Safe,
        SafetyVerdict::PossiblyUnsafe | SafetyVerdict::Unsafe => GroundTruth::Dangerous,
    }
}

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════════════╗");
    println!("║  GuardPharma — Real Polypharmacy Verification Benchmark              ║");
    println!("║  10 scenarios × (2-tier pipeline vs pairwise CYP-overlap baseline)   ║");
    println!("╚═══════════════════════════════════════════════════════════════════════╝\n");

    let all_scenarios = scenarios();
    let pipeline = VerificationPipeline::with_defaults();
    let config = pipeline.config().clone();

    println!("Pipeline config:");
    println!("  Tier1 promotion threshold: {:.1}", config.tier1_promotion_threshold);
    println!("  Simulation horizon: {:.0}h", config.simulation_horizon_hours);
    println!("  Tier1 time step: {:.1}h, Tier2 time step: {:.2}h", config.tier1_time_step_hours, config.tier2_time_step_hours);
    println!("  Toxic multiplier: {:.1}×", config.toxic_multiplier);
    println!();

    let mut results: Vec<ScenarioResult> = Vec::new();

    // Header
    println!("{:<45} {:>5} {:>10} {:>8} {:>8} {:>8}  {:>6}  {:>7}",
             "Scenario", "Drugs", "Truth", "GP-verd", "GP-ms", "BL-μs", "GP-ok", "BL-ok");
    println!("{}", "─".repeat(110));

    for sc in &all_scenarios {
        let patient = build_patient(sc.meds.clone());

        // ── Run GuardPharma pipeline ─────────────────────────────────
        let gp_start = Instant::now();
        let gp_result = pipeline.run(&patient);
        let _gp_elapsed = gp_start.elapsed();

        let gp_verdict = gp_result.verdict();
        let gp_correct = classify_verdict(gp_verdict) == sc.ground_truth;

        // ── Run baseline ─────────────────────────────────────────────
        let bl_start = Instant::now();
        let baseline = BaselineLookup::new(&sc.meds);
        let (bl_flagged, bl_total, bl_dangerous) = baseline.check();
        let bl_elapsed = bl_start.elapsed();
        let bl_correct = if sc.ground_truth == GroundTruth::Dangerous {
            bl_dangerous
        } else {
            !bl_dangerous
        };

        let sr = ScenarioResult {
            name: sc.name.to_string(),
            n_drugs: sc.meds.len(),
            ground_truth: sc.ground_truth,
            gp_verdict,
            gp_conflicts: gp_result.statistics.confirmed_conflicts,
            gp_promoted: gp_result.statistics.pairs_promoted_to_tier2,
            gp_tier1_ms: gp_result.statistics.tier1_duration_ms,
            gp_tier2_ms: gp_result.statistics.tier2_duration_ms,
            gp_total_ms: gp_result.statistics.total_duration_ms,
            gp_correct,
            bl_flagged,
            bl_total_pairs: bl_total,
            bl_verdict_dangerous: bl_dangerous,
            bl_correct,
            bl_time_us: bl_elapsed.as_micros() as u64,
        };

        let truth_str = match sc.ground_truth {
            GroundTruth::Dangerous => "DANGER",
            GroundTruth::Safe => "SAFE",
        };
        let gp_v = match gp_verdict {
            SafetyVerdict::Safe => "Safe",
            SafetyVerdict::PossiblySafe => "PossSafe",
            SafetyVerdict::PossiblyUnsafe => "PossUns",
            SafetyVerdict::Unsafe => "Unsafe",
        };
        let gp_ok = if gp_correct { "  ✓" } else { "  ✗" };
        let bl_ok = if bl_correct { "  ✓" } else { "  ✗" };

        println!("{:<45} {:>5} {:>10} {:>8} {:>8} {:>8}  {:>6}  {:>7}",
                 sc.name, sr.n_drugs, truth_str, gp_v,
                 sr.gp_total_ms, sr.bl_time_us, gp_ok, bl_ok);

        results.push(sr);
    }

    println!("{}", "─".repeat(110));
    println!();

    // ═══════════════════════════════════════════════════════════════════
    // Aggregate metrics
    // ═══════════════════════════════════════════════════════════════════

    let n_total = results.len();
    let n_dangerous = results.iter().filter(|r| r.ground_truth == GroundTruth::Dangerous).count();
    let n_safe = n_total - n_dangerous;

    // GuardPharma metrics
    let gp_tp = results.iter().filter(|r| r.ground_truth == GroundTruth::Dangerous && r.gp_correct).count();
    let gp_tn = results.iter().filter(|r| r.ground_truth == GroundTruth::Safe && r.gp_correct).count();
    let gp_fp = results.iter().filter(|r| r.ground_truth == GroundTruth::Safe && !r.gp_correct).count();
    let gp_fn = results.iter().filter(|r| r.ground_truth == GroundTruth::Dangerous && !r.gp_correct).count();

    // Baseline metrics
    let bl_tp = results.iter().filter(|r| r.ground_truth == GroundTruth::Dangerous && r.bl_correct).count();
    let bl_tn = results.iter().filter(|r| r.ground_truth == GroundTruth::Safe && r.bl_correct).count();
    let bl_fp = results.iter().filter(|r| r.ground_truth == GroundTruth::Safe && !r.bl_correct).count();
    let bl_fn = results.iter().filter(|r| r.ground_truth == GroundTruth::Dangerous && !r.bl_correct).count();

    let precision = |tp: usize, fp: usize| -> f64 {
        if tp + fp == 0 { 0.0 } else { tp as f64 / (tp + fp) as f64 }
    };
    let recall = |tp: usize, fn_: usize| -> f64 {
        if tp + fn_ == 0 { 0.0 } else { tp as f64 / (tp + fn_) as f64 }
    };
    let f1 = |p: f64, r: f64| -> f64 {
        if p + r == 0.0 { 0.0 } else { 2.0 * p * r / (p + r) }
    };

    let gp_prec = precision(gp_tp, gp_fp);
    let gp_rec = recall(gp_tp, gp_fn);
    let gp_f1 = f1(gp_prec, gp_rec);

    let bl_prec = precision(bl_tp, bl_fp);
    let bl_rec = recall(bl_tp, bl_fn);
    let bl_f1 = f1(bl_prec, bl_rec);

    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║                    AGGREGATE RESULTS                         ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    println!("Scenarios: {} total ({} dangerous, {} safe)\n", n_total, n_dangerous, n_safe);

    println!("                     GuardPharma    Baseline (CYP overlap)");
    println!("  ──────────────────────────────────────────────────────────");
    println!("  True Positives      {:>6}             {:>6}", gp_tp, bl_tp);
    println!("  True Negatives      {:>6}             {:>6}", gp_tn, bl_tn);
    println!("  False Positives     {:>6}             {:>6}", gp_fp, bl_fp);
    println!("  False Negatives     {:>6}             {:>6}", gp_fn, bl_fn);
    println!("  ──────────────────────────────────────────────────────────");
    println!("  Precision           {:>6.1}%            {:>6.1}%", gp_prec * 100.0, bl_prec * 100.0);
    println!("  Recall              {:>6.1}%            {:>6.1}%", gp_rec * 100.0, bl_rec * 100.0);
    println!("  F1 Score            {:>6.3}             {:>6.3}", gp_f1, bl_f1);
    println!("  Accuracy            {:>6.1}%            {:>6.1}%",
             (gp_tp + gp_tn) as f64 / n_total as f64 * 100.0,
             (bl_tp + bl_tn) as f64 / n_total as f64 * 100.0);
    println!();

    // Timing
    let gp_total_ms: u64 = results.iter().map(|r| r.gp_total_ms).sum();
    let gp_tier1_ms: u64 = results.iter().map(|r| r.gp_tier1_ms).sum();
    let gp_tier2_ms: u64 = results.iter().map(|r| r.gp_tier2_ms).sum();
    let bl_total_us: u64 = results.iter().map(|r| r.bl_time_us).sum();
    let gp_avg_ms = gp_total_ms as f64 / n_total as f64;
    let bl_avg_us = bl_total_us as f64 / n_total as f64;

    println!("Timing:");
    println!("  GuardPharma total:  {}ms  (Tier1: {}ms, Tier2: {}ms)",
             gp_total_ms, gp_tier1_ms, gp_tier2_ms);
    println!("  GuardPharma avg:    {:.1}ms per scenario", gp_avg_ms);
    println!("  Baseline total:     {}μs", bl_total_us);
    println!("  Baseline avg:       {:.1}μs per scenario", bl_avg_us);
    println!("  Speedup (baseline): {:.0}× faster (but less precise)",
             if bl_avg_us > 0.0 { gp_avg_ms * 1000.0 / bl_avg_us } else { 0.0 });
    println!();

    // Detail dump
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║                    DETAILED RESULTS                          ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    for (i, (sc, r)) in all_scenarios.iter().zip(results.iter()).enumerate() {
        let mark = if r.gp_correct { "✓" } else { "✗" };
        println!("{}. [{}] {} ({})", i + 1, mark, sc.name, sc.description);
        println!("   Ground truth: {:?}", sc.ground_truth);
        println!("   Expected mechanism: {}", sc.expected_mechanism);
        println!("   GuardPharma: verdict={:?}, conflicts={}, promoted={}, time={}ms",
                 r.gp_verdict, r.gp_conflicts, r.gp_promoted, r.gp_total_ms);
        println!("   Baseline: flagged={}/{} pairs, verdict={}, time={}μs",
                 r.bl_flagged, r.bl_total_pairs,
                 if r.bl_verdict_dangerous { "DANGER" } else { "SAFE" },
                 r.bl_time_us);
        if !r.gp_correct {
            println!("   ⚠ GuardPharma MISCLASSIFIED this scenario");
        }
        if !r.bl_correct {
            println!("   ⚠ Baseline MISCLASSIFIED this scenario");
        }
        println!();
    }

    // ── False positive analysis ──────────────────────────────────────
    let gp_fp_scenarios: Vec<&ScenarioResult> = results.iter()
        .filter(|r| r.ground_truth == GroundTruth::Safe && !r.gp_correct)
        .collect();
    let bl_fp_scenarios: Vec<&ScenarioResult> = results.iter()
        .filter(|r| r.ground_truth == GroundTruth::Safe && !r.bl_correct)
        .collect();

    println!("False Positive Analysis:");
    println!("  GuardPharma FP rate on safe combos: {}/{} ({:.0}%)",
             gp_fp_scenarios.len(), n_safe,
             if n_safe > 0 { gp_fp_scenarios.len() as f64 / n_safe as f64 * 100.0 } else { 0.0 });
    for r in &gp_fp_scenarios {
        println!("    - {}", r.name);
    }
    println!("  Baseline FP rate on safe combos: {}/{} ({:.0}%)",
             bl_fp_scenarios.len(), n_safe,
             if n_safe > 0 { bl_fp_scenarios.len() as f64 / n_safe as f64 * 100.0 } else { 0.0 });
    for r in &bl_fp_scenarios {
        println!("    - {}", r.name);
    }
    println!();

    // ── JSON output ──────────────────────────────────────────────────
    println!("───── JSON RESULTS (for pipeline ingestion) ─────");
    let json = serde_json::json!({
        "benchmark": "real_polypharmacy_verification",
        "scenarios": n_total,
        "dangerous_scenarios": n_dangerous,
        "safe_scenarios": n_safe,
        "guardpharma": {
            "true_positives": gp_tp,
            "true_negatives": gp_tn,
            "false_positives": gp_fp,
            "false_negatives": gp_fn,
            "precision": gp_prec,
            "recall": gp_rec,
            "f1_score": gp_f1,
            "accuracy": (gp_tp + gp_tn) as f64 / n_total as f64,
            "total_time_ms": gp_total_ms,
            "tier1_time_ms": gp_tier1_ms,
            "tier2_time_ms": gp_tier2_ms,
            "avg_time_ms": gp_avg_ms,
        },
        "baseline_cyp_overlap": {
            "true_positives": bl_tp,
            "true_negatives": bl_tn,
            "false_positives": bl_fp,
            "false_negatives": bl_fn,
            "precision": bl_prec,
            "recall": bl_rec,
            "f1_score": bl_f1,
            "accuracy": (bl_tp + bl_tn) as f64 / n_total as f64,
            "total_time_us": bl_total_us,
            "avg_time_us": bl_avg_us,
        },
        "per_scenario": results.iter().zip(all_scenarios.iter()).map(|(r, sc)| {
            serde_json::json!({
                "name": r.name,
                "n_drugs": r.n_drugs,
                "ground_truth": format!("{:?}", r.ground_truth),
                "guardpharma_verdict": format!("{:?}", r.gp_verdict),
                "guardpharma_correct": r.gp_correct,
                "guardpharma_conflicts": r.gp_conflicts,
                "guardpharma_promoted": r.gp_promoted,
                "guardpharma_time_ms": r.gp_total_ms,
                "baseline_flagged": r.bl_flagged,
                "baseline_total_pairs": r.bl_total_pairs,
                "baseline_correct": r.bl_correct,
                "baseline_time_us": r.bl_time_us,
                "expected_mechanism": sc.expected_mechanism,
            })
        }).collect::<Vec<_>>(),
    });
    println!("{}", serde_json::to_string_pretty(&json).unwrap());
}
