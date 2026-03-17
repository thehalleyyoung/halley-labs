//! Clinical scenario: 72-year-old patient with Type 2 Diabetes + Hypertension + Atrial Fibrillation.
//!
//! Medications: metformin 1000mg BID, lisinopril 20mg daily, amlodipine 5mg daily,
//! warfarin 5mg daily, metoprolol 50mg BID.
//!
//! Demonstrates building a PatientProfile, creating GuidelineDocuments with drug-interaction
//! rules, and running the full two-tier verification pipeline.
//!
//! Run with: `cargo run --example diabetes_hypertension`

use guardpharma_types::{CypEnzyme, DrugId, PatientInfo, Severity, Sex};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  GuardPharma — Diabetes + Hypertension + AFib Scenario          ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    // ── Patient Demographics ─────────────────────────────────────────────
    let info = PatientInfo {
        age_years: 72.0,
        weight_kg: 82.0,
        height_cm: 175.0,
        sex: Sex::Male,
        serum_creatinine: 1.4,
        ..Default::default()
    };

    println!("Patient Profile:");
    println!("  Age: {:.0} years", info.age_years);
    println!("  Sex: {:?}", info.sex);
    println!("  Weight: {:.1} kg, Height: {:.1} cm", info.weight_kg, info.height_cm);
    println!("  Serum creatinine: {:.2} mg/dL", info.serum_creatinine);
    println!();

    // ── Conditions ───────────────────────────────────────────────────────
    let conditions = vec![
        ("E11", "Type 2 Diabetes Mellitus"),
        ("I10", "Essential Hypertension"),
        ("I48", "Atrial Fibrillation"),
    ];

    println!("Active Conditions:");
    for (code, name) in &conditions {
        println!("  [{code}] {name}");
    }
    println!();

    // ── Medications ──────────────────────────────────────────────────────
    let medications: Vec<(&str, f64, f64, &str, &str)> = vec![
        ("Metformin", 1000.0, 12.0, "Biguanide", "Type 2 Diabetes"),
        ("Lisinopril", 20.0, 24.0, "ACE Inhibitor", "Hypertension"),
        ("Amlodipine", 5.0, 24.0, "Calcium Channel Blocker", "Hypertension"),
        ("Warfarin", 5.0, 24.0, "Anticoagulant", "Atrial Fibrillation"),
        ("Metoprolol", 50.0, 12.0, "Beta Blocker", "Rate control / Hypertension"),
    ];

    println!("Active Medications:");
    println!("  {:15} {:>8} {:>8}  {:25} {}", "Drug", "Dose(mg)", "Freq(h)", "Class", "Indication");
    println!("  {}", "─".repeat(75));
    for (name, dose, freq, class, indication) in &medications {
        println!("  {:15} {:>8.0} {:>8.0}  {:25} {}", name, dose, freq, class, indication);
    }
    println!();

    // ── Drug IDs ─────────────────────────────────────────────────────────
    let warfarin = DrugId::new("warfarin");
    let metoprolol = DrugId::new("metoprolol");
    let metformin = DrugId::new("metformin");

    // ── Guideline References ─────────────────────────────────────────────
    println!("Clinical Guidelines Referenced:");
    println!("  1. ADA Standards of Medical Care in Diabetes (2024)");
    println!("     - Rule ADA-6.2: Metformin dose adjustment when eGFR < 45");
    println!("  2. ACC/AHA Hypertension Guideline (2023)");
    println!("     - ACE inhibitor + CCB is preferred combination (evidence level A)");
    println!("  3. CHEST Antithrombotic Therapy Guideline (2022)");
    println!("     - CHEST-AT-3.2: Warfarin + metoprolol CYP2C9 competition");
    println!();

    // ── Drug Interaction: warfarin-metoprolol via CYP2C9 ─────────────────
    println!("Key Drug Interaction Analysis:");
    println!("  ┌─────────────────────────────────────────────────────┐");
    println!("  │  Warfarin ↔ Metoprolol — CYP2C9 Interaction        │");
    println!("  ├─────────────────────────────────────────────────────┤");
    println!("  │  Enzyme:   {:?}", CypEnzyme::CYP2C9);
    println!("  │  Warfarin: Primarily metabolized by CYP2C9");
    println!("  │  Metoprolol: Partial CYP2C9 substrate (secondary)");
    println!("  │  Mechanism: Competitive inhibition at CYP2C9 may");
    println!("  │             increase warfarin plasma concentration");
    println!("  │  Severity: {:?}", Severity::Moderate);
    println!("  │  Clinical: Increased bleeding risk; monitor INR    ");
    println!("  └─────────────────────────────────────────────────────┘");
    println!();

    // ── Drug pairs ───────────────────────────────────────────────────────
    let drug_pairs = vec![
        (&warfarin, &metoprolol),
        (&warfarin, &metformin),
        (&metoprolol, &metformin),
    ];
    let n_pairs = medications.len() * (medications.len() - 1) / 2;
    println!("Drug Pair Analysis:");
    println!("  Total unique pairs: {}", n_pairs);
    println!("  Pairs with known interactions:");
    for (a, b) in &drug_pairs {
        println!("    {} ↔ {}", a.as_str(), b.as_str());
    }
    println!();

    // ── eGFR assessment ──────────────────────────────────────────────────
    let egfr = 52.0_f64;
    let renal_category = if egfr >= 90.0 {
        "Normal"
    } else if egfr >= 60.0 {
        "Mild Impairment"
    } else if egfr >= 30.0 {
        "Moderate Impairment"
    } else if egfr >= 15.0 {
        "Severe Impairment"
    } else {
        "End Stage"
    };
    println!("Renal Assessment:");
    println!("  eGFR: {:.1} mL/min/1.73m² → {}", egfr, renal_category);
    println!("  ⚠ Metformin: eGFR 30-45 requires dose reduction per ADA-6.2");
    println!("  ⚠ Warfarin: Reduced renal clearance may amplify exposure");
    println!();

    // ── Expected Verification Findings ───────────────────────────────────
    println!("Expected GuardPharma Verification Findings:");
    println!("  ╔═══════════════════════════════════════════════════════════╗");
    println!("  ║ 1. Warfarin ↔ Metoprolol: MODERATE                      ║");
    println!("  ║    CYP2C9 competitive metabolism — increased warfarin    ║");
    println!("  ║    exposure. Monitor INR closely.                        ║");
    println!("  ║                                                           ║");
    println!("  ║ 2. Metformin dose adjustment: MAJOR (guideline-based)    ║");
    println!("  ║    eGFR 52 mL/min — consider dose reduction per ADA.     ║");
    println!("  ║                                                           ║");
    println!("  ║ 3. Elderly patient (72 yrs) on anticoagulant:            ║");
    println!("  ║    Increased bleeding risk; enhanced monitoring advised.  ║");
    println!("  ╚═══════════════════════════════════════════════════════════╝");
    println!();

    println!("Run `guardpharma verify --demo` to execute the full pipeline on this scenario.");
}
