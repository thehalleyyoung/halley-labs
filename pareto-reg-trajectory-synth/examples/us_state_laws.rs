//! # US State-Level AI Legislation Analysis
//!
//! This example demonstrates RegSynth's multi-jurisdictional capability by
//! modeling obligations from three US state AI laws:
//!
//! ## California (SB 1047 / AB 2013)
//! - Frontier model safety evaluations
//! - Pre-deployment safety testing
//! - Kill-switch / shutdown capability requirement
//! - Transparency reporting for generative AI (AB 2013)
//!
//! ## Colorado (SB 24-205, AI Consumer Protections)
//! - Algorithmic discrimination prevention for "high-risk" decisions
//! - Risk management policy requirement
//! - Impact assessments before deployment
//! - Annual compliance reporting to AG
//! - Consumer notification of AI-driven consequential decisions
//!
//! ## Illinois (BIPA + HB 3773 / AI Video Interview Act)
//! - Biometric information consent (BIPA §15)
//! - AI video interview disclosure and consent
//! - Data destruction requirements (BIPA §15(a))
//! - Bias audit for AI hiring tools
//!
//! ## Approach
//!
//! We model each jurisdiction's obligations, assign formalizability grades,
//! build a jurisdiction lattice, and identify the obligation intersection
//! for a company operating in all three states.

use regsynth_types::{
    ArticleRef, FormalizabilityGrade, Jurisdiction, ObligationKind, RiskLevel,
    TemporalInterval,
};
use regsynth_temporal::Obligation;
use regsynth_pareto::{
    CostVector as ParetoCostVector, ParetoFrontier, dominates,
};

use chrono::NaiveDate;

/// Build California AI obligations.
fn california_obligations() -> Vec<Obligation> {
    let ca = Jurisdiction::new("US-CA");
    vec![
        Obligation::new(
            "ca-sb1047-safety-eval",
            ObligationKind::Obligation,
            ca.clone(),
            "Pre-deployment safety evaluation for frontier models",
        )
        .with_article_ref(ArticleRef {
            framework: "CA-SB-1047".into(),
            article: "22602".into(),
            paragraph: Some("a".into()),
        })
        .with_risk_level(RiskLevel::High)
        .with_grade(FormalizabilityGrade::F3),

        Obligation::new(
            "ca-sb1047-killswitch",
            ObligationKind::Obligation,
            ca.clone(),
            "Shutdown capability for covered models",
        )
        .with_article_ref(ArticleRef {
            framework: "CA-SB-1047".into(),
            article: "22602".into(),
            paragraph: Some("b".into()),
        })
        .with_risk_level(RiskLevel::High)
        .with_grade(FormalizabilityGrade::F1),

        Obligation::new(
            "ca-ab2013-genai-transparency",
            ObligationKind::Obligation,
            ca.clone(),
            "Generative AI transparency: disclose AI-generated content",
        )
        .with_article_ref(ArticleRef {
            framework: "CA-AB-2013".into(),
            article: "1".into(),
            paragraph: None,
        })
        .with_risk_level(RiskLevel::Limited)
        .with_grade(FormalizabilityGrade::F2),
    ]
}

/// Build Colorado AI obligations.
fn colorado_obligations() -> Vec<Obligation> {
    let co = Jurisdiction::new("US-CO");
    vec![
        Obligation::new(
            "co-sb205-risk-mgmt",
            ObligationKind::Obligation,
            co.clone(),
            "Risk management policy for high-risk AI systems",
        )
        .with_article_ref(ArticleRef {
            framework: "CO-SB-24-205".into(),
            article: "6-1-1702".into(),
            paragraph: Some("1".into()),
        })
        .with_risk_level(RiskLevel::High)
        .with_grade(FormalizabilityGrade::F2),

        Obligation::new(
            "co-sb205-impact-assessment",
            ObligationKind::Obligation,
            co.clone(),
            "Impact assessment before deployment of high-risk AI",
        )
        .with_article_ref(ArticleRef {
            framework: "CO-SB-24-205".into(),
            article: "6-1-1702".into(),
            paragraph: Some("2".into()),
        })
        .with_risk_level(RiskLevel::High)
        .with_grade(FormalizabilityGrade::F3),

        Obligation::new(
            "co-sb205-consumer-notice",
            ObligationKind::Obligation,
            co.clone(),
            "Notify consumers of AI-driven consequential decisions",
        )
        .with_article_ref(ArticleRef {
            framework: "CO-SB-24-205".into(),
            article: "6-1-1703".into(),
            paragraph: None,
        })
        .with_risk_level(RiskLevel::High)
        .with_grade(FormalizabilityGrade::F2),

        Obligation::new(
            "co-sb205-annual-report",
            ObligationKind::Obligation,
            co.clone(),
            "Annual compliance reporting to Attorney General",
        )
        .with_article_ref(ArticleRef {
            framework: "CO-SB-24-205".into(),
            article: "6-1-1704".into(),
            paragraph: None,
        })
        .with_risk_level(RiskLevel::High)
        .with_grade(FormalizabilityGrade::F1),
    ]
}

/// Build Illinois AI obligations (BIPA + AI Video Interview Act).
fn illinois_obligations() -> Vec<Obligation> {
    let il = Jurisdiction::new("US-IL");
    vec![
        Obligation::new(
            "il-bipa-consent",
            ObligationKind::Obligation,
            il.clone(),
            "Written consent before collecting biometric data (BIPA §15(b))",
        )
        .with_article_ref(ArticleRef {
            framework: "IL-BIPA".into(),
            article: "15".into(),
            paragraph: Some("b".into()),
        })
        .with_risk_level(RiskLevel::High)
        .with_grade(FormalizabilityGrade::F1),

        Obligation::new(
            "il-bipa-destruction",
            ObligationKind::Obligation,
            il.clone(),
            "Destroy biometric data when purpose fulfilled (BIPA §15(a))",
        )
        .with_article_ref(ArticleRef {
            framework: "IL-BIPA".into(),
            article: "15".into(),
            paragraph: Some("a".into()),
        })
        .with_risk_level(RiskLevel::High)
        .with_grade(FormalizabilityGrade::F2),

        Obligation::new(
            "il-aivi-disclosure",
            ObligationKind::Obligation,
            il.clone(),
            "Disclose use of AI in video interviews to applicants",
        )
        .with_article_ref(ArticleRef {
            framework: "IL-AIVI-Act".into(),
            article: "5".into(),
            paragraph: None,
        })
        .with_risk_level(RiskLevel::Limited)
        .with_grade(FormalizabilityGrade::F1),

        Obligation::new(
            "il-aivi-bias-audit",
            ObligationKind::Obligation,
            il.clone(),
            "Annual bias audit for AI hiring tools",
        )
        .with_article_ref(ArticleRef {
            framework: "IL-AIVI-Act".into(),
            article: "10".into(),
            paragraph: None,
        })
        .with_risk_level(RiskLevel::High)
        .with_grade(FormalizabilityGrade::F3),
    ]
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  RegSynth — US State-Level AI Legislation Analysis         ║");
    println!("║  Jurisdictions: California, Colorado, Illinois             ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let ca_obls = california_obligations();
    let co_obls = colorado_obligations();
    let il_obls = illinois_obligations();

    let all_obligations: Vec<&Obligation> = ca_obls.iter()
        .chain(co_obls.iter())
        .chain(il_obls.iter())
        .collect();

    // 1. Per-jurisdiction summary
    println!("📋 Obligations by Jurisdiction:\n");

    for (name, obls) in [("California", &ca_obls), ("Colorado", &co_obls), ("Illinois", &il_obls)] {
        println!("  🏛  {} ({} obligations)", name, obls.len());
        for obl in obls {
            let art = obl.article_ref.as_ref()
                .map(|a| format!("{}", a))
                .unwrap_or_else(|| "N/A".into());
            println!("     [{kind}] {id} (Grade: {grade}, Article: {art})",
                kind = obl.kind, id = obl.id, grade = obl.grade);
        }
        println!();
    }

    // 2. Jurisdiction lattice
    let us = Jurisdiction::new("US");
    let ca = Jurisdiction::new("US-CA");
    let co = Jurisdiction::new("US-CO");
    let il = Jurisdiction::new("US-IL");

    println!("🌐 Jurisdiction Lattice:");
    println!("  US is parent of US-CA: {}", us.is_parent_of(&ca));
    println!("  US is parent of US-CO: {}", us.is_parent_of(&co));
    println!("  US is parent of US-IL: {}", us.is_parent_of(&il));
    println!("  US-CA is parent of US-CO: {}", ca.is_parent_of(&co));
    println!();

    // 3. Formalizability analysis
    println!("📊 Formalizability Distribution:");
    for grade_num in 1..=5 {
        let grade = FormalizabilityGrade::from_u8(grade_num).unwrap();
        let count = all_obligations.iter()
            .filter(|o| o.grade == grade)
            .count();
        let bar = "█".repeat(count * 4);
        println!("  {} │ {} ({})", grade, bar, count);
    }
    println!();

    // 4. Cross-state compliance cost analysis
    println!("💰 Cross-State Compliance Cost Analysis (Pareto Frontier):");

    let strategies = vec![
        ("Minimal viable (each state separately)", ParetoCostVector::regulatory(200_000.0, 14.0, 0.15, 40.0)),
        ("Shared framework (common controls)", ParetoCostVector::regulatory(350_000.0, 10.0, 0.08, 55.0)),
        ("Full automation (CI/CD compliance)", ParetoCostVector::regulatory(600_000.0, 6.0, 0.03, 85.0)),
        ("Outsourced compliance", ParetoCostVector::regulatory(450_000.0, 4.0, 0.10, 30.0)),
    ];

    let mut frontier: ParetoFrontier<String> = ParetoFrontier::new(4);
    for (name, cv) in &strategies {
        let added = frontier.add_point(name.to_string(), cv.clone());
        let status = if added { "✅ non-dominated" } else { "❌ dominated" };
        println!("  {} — {} : {}", status, name, cv);
    }

    println!("\n  Frontier size: {} strategies", frontier.size());

    // 5. Risk summary
    let high_risk_count = all_obligations.iter()
        .filter(|o| o.risk_level == Some(RiskLevel::High))
        .count();
    println!("\n⚠️  Risk Summary:");
    println!("  Total obligations: {}", all_obligations.len());
    println!("  High-risk: {}", high_risk_count);
    println!("  Limited-risk: {}", all_obligations.len() - high_risk_count);

    println!("\n✅ Multi-jurisdictional analysis complete.");
}
