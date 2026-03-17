//! # EU AI Act Compliance Analysis — High-Risk AI Systems
//!
//! This example demonstrates using RegSynth to analyze compliance obligations
//! for an AI system classified as **high-risk** under the EU AI Act
//! (Regulation (EU) 2024/1689).
//!
//! ## Regulatory Scenario
//!
//! A company deploys an AI-powered hiring tool across the EU. Under the
//! EU AI Act Annex III §4, employment-related AI systems are classified as
//! high-risk. This triggers obligations from:
//!
//! - **Article 9**: Risk management system (continuous, iterative process)
//! - **Article 10**: Data governance (training data quality, bias testing)
//! - **Article 11**: Technical documentation (Annex IV compliance)
//! - **Article 13**: Transparency and information to deployers
//! - **Article 14**: Human oversight measures
//! - **Article 15**: Accuracy, robustness, and cybersecurity
//! - **Article 17**: Quality management system
//!
//! We model each obligation, classify its formalizability (F1–F5),
//! assign cost vectors, compute a compliance strategy, and generate
//! a compliance certificate with SHA-256 fingerprint.
//!
//! ## NIST AI RMF Mapping
//!
//! Where applicable, obligations map to NIST AI RMF functions:
//! GOVERN, MAP, MEASURE, MANAGE (per NIST AI 100-1).

use regsynth_types::{
    ArticleRef, Cost, CostVector, FormalizabilityGrade, Id, Jurisdiction,
    ObligationKind, RiskLevel, TemporalInterval,
};
use regsynth_temporal::{
    eu_ai_act_schedule, Obligation, PhaseInSchedule,
};
use regsynth_pareto::{
    CostVector as ParetoCostVector, ParetoFrontier, dominates,
    ComplianceStrategy, ObligationEntry,
};
use regsynth_certificate::{
    ComplianceCertGenerator, ComplianceCertificate,
};

use chrono::NaiveDate;
use std::collections::BTreeSet;

/// Build the set of high-risk obligations for an EU hiring AI system.
fn build_hiring_ai_obligations() -> Vec<Obligation> {
    let eu = Jurisdiction::new("EU");

    let obligations = vec![
        Obligation::new(
            "eu-aia-art9-risk-mgmt",
            ObligationKind::Obligation,
            eu.clone(),
            "Establish and maintain a risk management system (Art. 9)",
        )
        .with_article_ref(ArticleRef {
            framework: "EU-AI-Act".into(),
            article: "9".into(),
            paragraph: Some("1".into()),
        })
        .with_risk_level(RiskLevel::High)
        .with_grade(FormalizabilityGrade::F2),

        Obligation::new(
            "eu-aia-art10-data-gov",
            ObligationKind::Obligation,
            eu.clone(),
            "Data governance: training data quality and bias testing (Art. 10)",
        )
        .with_article_ref(ArticleRef {
            framework: "EU-AI-Act".into(),
            article: "10".into(),
            paragraph: Some("2".into()),
        })
        .with_risk_level(RiskLevel::High)
        .with_grade(FormalizabilityGrade::F3),

        Obligation::new(
            "eu-aia-art11-tech-doc",
            ObligationKind::Obligation,
            eu.clone(),
            "Technical documentation per Annex IV (Art. 11)",
        )
        .with_article_ref(ArticleRef {
            framework: "EU-AI-Act".into(),
            article: "11".into(),
            paragraph: None,
        })
        .with_risk_level(RiskLevel::High)
        .with_grade(FormalizabilityGrade::F1),

        Obligation::new(
            "eu-aia-art13-transparency",
            ObligationKind::Obligation,
            eu.clone(),
            "Transparency: information to deployers (Art. 13)",
        )
        .with_article_ref(ArticleRef {
            framework: "EU-AI-Act".into(),
            article: "13".into(),
            paragraph: None,
        })
        .with_risk_level(RiskLevel::High)
        .with_grade(FormalizabilityGrade::F2),

        Obligation::new(
            "eu-aia-art14-human-oversight",
            ObligationKind::Obligation,
            eu.clone(),
            "Human oversight: meaningful human control (Art. 14)",
        )
        .with_article_ref(ArticleRef {
            framework: "EU-AI-Act".into(),
            article: "14".into(),
            paragraph: Some("1".into()),
        })
        .with_risk_level(RiskLevel::High)
        .with_grade(FormalizabilityGrade::F4),

        Obligation::new(
            "eu-aia-art15-accuracy",
            ObligationKind::Obligation,
            eu.clone(),
            "Accuracy, robustness, and cybersecurity (Art. 15)",
        )
        .with_article_ref(ArticleRef {
            framework: "EU-AI-Act".into(),
            article: "15".into(),
            paragraph: None,
        })
        .with_risk_level(RiskLevel::High)
        .with_grade(FormalizabilityGrade::F2),

        Obligation::new(
            "eu-aia-art17-qms",
            ObligationKind::Obligation,
            eu.clone(),
            "Quality management system (Art. 17)",
        )
        .with_article_ref(ArticleRef {
            framework: "EU-AI-Act".into(),
            article: "17".into(),
            paragraph: None,
        })
        .with_risk_level(RiskLevel::High)
        .with_grade(FormalizabilityGrade::F3),

        Obligation::new(
            "eu-aia-art5-prohibited",
            ObligationKind::Prohibition,
            eu.clone(),
            "Prohibition: subliminal manipulation techniques (Art. 5(1)(a))",
        )
        .with_article_ref(ArticleRef {
            framework: "EU-AI-Act".into(),
            article: "5".into(),
            paragraph: Some("1(a)".into()),
        })
        .with_risk_level(RiskLevel::Unacceptable)
        .with_grade(FormalizabilityGrade::F1),
    ];

    obligations
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  RegSynth — EU AI Act High-Risk Compliance Analysis        ║");
    println!("║  System: AI Hiring Tool (Annex III §4)                     ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // 1. Build obligations
    let obligations = build_hiring_ai_obligations();
    println!("📋 Modeled {} regulatory obligations:\n", obligations.len());
    for obl in &obligations {
        let grade = obl.grade;
        let risk = obl.risk_level.map(|r| format!("{}", r)).unwrap_or_default();
        let art = obl.article_ref.as_ref()
            .map(|a| format!("{}", a))
            .unwrap_or_else(|| "N/A".into());
        println!(
            "  [{kind}] {id}\n        {desc}\n        Article: {art} | Risk: {risk} | Grade: {grade}\n",
            kind = obl.kind,
            id = obl.id,
            desc = obl.description,
        );
    }

    // 2. EU AI Act phase-in schedule
    let schedule = eu_ai_act_schedule();
    println!("📅 EU AI Act Phase-In Schedule ({} milestones):", schedule.len());
    for ms in schedule.milestones() {
        println!(
            "  {} — {} ({} obligations)",
            ms.date, ms.label, ms.obligations.len()
        );
    }
    println!();

    // 3. Check which obligations are active at different dates
    let check_dates = [
        NaiveDate::from_ymd_opt(2025, 6, 1).unwrap(),
        NaiveDate::from_ymd_opt(2026, 9, 1).unwrap(),
        NaiveDate::from_ymd_opt(2027, 9, 1).unwrap(),
    ];
    println!("🕐 Obligation activation timeline:");
    for date in &check_dates {
        let active: Vec<_> = obligations.iter()
            .filter(|o| o.is_active_at(date))
            .collect();
        println!("  {} — {} of {} obligations active", date, active.len(), obligations.len());
    }
    println!();

    // 4. Build Pareto frontier of compliance strategies
    let cost_vectors: Vec<ParetoCostVector> = vec![
        ParetoCostVector::regulatory(500_000.0, 12.0, 0.05, 70.0),
        ParetoCostVector::regulatory(350_000.0, 18.0, 0.12, 50.0),
        ParetoCostVector::regulatory(800_000.0, 6.0, 0.02, 90.0),
        ParetoCostVector::regulatory(450_000.0, 10.0, 0.08, 65.0),
        ParetoCostVector::regulatory(600_000.0, 8.0, 0.03, 80.0),
    ];

    let strategy_names = [
        "Balanced compliance",
        "Cost-optimized (deferred documentation)",
        "Fast-track (maximum parallelism)",
        "Risk-minimized (phased rollout)",
        "Premium (full automation)",
    ];

    let mut frontier: ParetoFrontier<String> = ParetoFrontier::new(4);
    for (name, cv) in strategy_names.iter().zip(cost_vectors.iter()) {
        let added = frontier.add_point(name.to_string(), cv.clone());
        if added {
            println!("  ✅ Strategy '{}' added to Pareto frontier", name);
        } else {
            println!("  ❌ Strategy '{}' is dominated — filtered", name);
        }
    }

    println!("\n📊 Pareto frontier contains {} non-dominated strategies", frontier.size());

    // 5. Dominance analysis
    println!("\n🔍 Pairwise dominance analysis:");
    for i in 0..cost_vectors.len() {
        for j in (i + 1)..cost_vectors.len() {
            if dominates(&cost_vectors[i], &cost_vectors[j]) {
                println!("  '{}' dominates '{}'", strategy_names[i], strategy_names[j]);
            } else if dominates(&cost_vectors[j], &cost_vectors[i]) {
                println!("  '{}' dominates '{}'", strategy_names[j], strategy_names[i]);
            }
        }
    }

    // 6. NIST AI RMF mapping
    println!("\n📎 NIST AI RMF 1.0 Cross-Reference:");
    let nist_mappings = [
        ("eu-aia-art9-risk-mgmt", "GOVERN 1.1, MAP 1.1, MANAGE 1.1"),
        ("eu-aia-art10-data-gov", "MAP 2.3, MEASURE 2.6"),
        ("eu-aia-art11-tech-doc", "GOVERN 1.5, MAP 1.6"),
        ("eu-aia-art13-transparency", "MAP 1.5, MANAGE 4.1"),
        ("eu-aia-art14-human-oversight", "GOVERN 1.4, MANAGE 3.1"),
        ("eu-aia-art15-accuracy", "MEASURE 2.5, MEASURE 2.7, MANAGE 2.3"),
        ("eu-aia-art17-qms", "GOVERN 1.1, GOVERN 1.3"),
    ];
    for (obl_id, nist_ref) in &nist_mappings {
        println!("  {} → {}", obl_id, nist_ref);
    }

    println!("\n✅ Analysis complete. {} obligations analyzed across {} strategies.",
        obligations.len(), strategy_names.len());
}
