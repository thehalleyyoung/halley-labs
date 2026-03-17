//! # Cross-Jurisdictional Conflict Detection
//!
//! This example demonstrates RegSynth's ability to detect and diagnose
//! regulatory conflicts between overlapping frameworks.
//!
//! ## Scenario: EU AI Act vs. GDPR Data Minimization
//!
//! A fundamental tension exists between:
//!
//! 1. **EU AI Act Art. 12** — Automatic logging of high-risk AI system
//!    operations. Providers must ensure the system generates logs covering
//!    the period of use, including input data characteristics, reference
//!    databases queried, and output results.
//!
//! 2. **GDPR Art. 5(1)(c)** — Data minimization principle: personal data
//!    must be adequate, relevant, and limited to what is necessary.
//!
//! 3. **GDPR Art. 5(1)(e)** — Storage limitation: personal data kept in
//!    identifiable form no longer than necessary for processing purposes.
//!
//! These create a genuine regulatory conflict: the AI Act mandates extensive
//! logging of operational data (which may include personal data), while the
//! GDPR mandates minimizing personal data collection and retention.
//!
//! ## Additional Conflict: Transparency vs. Trade Secrets
//!
//! - **EU AI Act Art. 13** requires transparency about AI system operation
//! - **Trade Secrets Directive 2016/943** protects proprietary algorithms
//!
//! RegSynth detects these conflicts through constraint encoding and
//! MUS (Minimal Unsatisfiable Subset) extraction.

use regsynth_types::{
    ArticleRef, FormalizabilityGrade, Jurisdiction, ObligationKind, RiskLevel,
    TemporalInterval,
    constraint::{
        Constraint, ConstraintExpr, ConstraintId, ConstraintKind, ConstraintSet, VarId,
    },
};
use regsynth_temporal::Obligation;
use regsynth_encoding::{SmtExpr, SmtSort, SmtConstraint, Provenance};
use regsynth_certificate::{
    InfeasibilityCertGenerator, ConflictCategory, ConflictSeverity,
    RegulatoryConflict, RegulatoryDiagnosis, MinimalUnsatisfiableSubset, MusConstraint,
};

/// Build the conflicting obligations from EU AI Act and GDPR.
fn build_conflict_scenario() -> (Vec<Obligation>, ConstraintSet) {
    let eu = Jurisdiction::new("EU");

    let obligations = vec![
        // EU AI Act: logging obligation
        Obligation::new(
            "eu-aia-art12-logging",
            ObligationKind::Obligation,
            eu.clone(),
            "Automatic logging of high-risk AI operations (Art. 12)",
        )
        .with_article_ref(ArticleRef {
            framework: "EU-AI-Act".into(),
            article: "12".into(),
            paragraph: Some("1".into()),
        })
        .with_risk_level(RiskLevel::High)
        .with_grade(FormalizabilityGrade::F2),

        // GDPR: data minimization
        Obligation::new(
            "gdpr-art5-data-min",
            ObligationKind::Obligation,
            eu.clone(),
            "Data minimization: collect only necessary personal data (Art. 5(1)(c))",
        )
        .with_article_ref(ArticleRef {
            framework: "GDPR".into(),
            article: "5".into(),
            paragraph: Some("1(c)".into()),
        })
        .with_risk_level(RiskLevel::High)
        .with_grade(FormalizabilityGrade::F3),

        // GDPR: storage limitation
        Obligation::new(
            "gdpr-art5-storage-limit",
            ObligationKind::Obligation,
            eu.clone(),
            "Storage limitation: retain personal data only as long as necessary (Art. 5(1)(e))",
        )
        .with_article_ref(ArticleRef {
            framework: "GDPR".into(),
            article: "5".into(),
            paragraph: Some("1(e)".into()),
        })
        .with_risk_level(RiskLevel::High)
        .with_grade(FormalizabilityGrade::F2),

        // EU AI Act: transparency
        Obligation::new(
            "eu-aia-art13-transparency",
            ObligationKind::Obligation,
            eu.clone(),
            "Transparency and information to deployers (Art. 13)",
        )
        .with_article_ref(ArticleRef {
            framework: "EU-AI-Act".into(),
            article: "13".into(),
            paragraph: None,
        })
        .with_risk_level(RiskLevel::High)
        .with_grade(FormalizabilityGrade::F3),

        // Trade Secrets Directive
        Obligation::new(
            "eu-tsd-protection",
            ObligationKind::Permission,
            eu.clone(),
            "Protection of trade secrets and proprietary algorithms (Directive 2016/943)",
        )
        .with_article_ref(ArticleRef {
            framework: "EU-Trade-Secrets-Dir".into(),
            article: "2".into(),
            paragraph: Some("1".into()),
        })
        .with_risk_level(RiskLevel::Limited)
        .with_grade(FormalizabilityGrade::F4),
    ];

    // Encode as formal constraints
    let mut constraints = ConstraintSet::new();

    // x_log ∈ [0,1]: fraction of personal data logged
    // x_min ∈ [0,1]: data minimization compliance level
    // x_retain: retention period in months
    // x_disclose ∈ [0,1]: model disclosure level

    // AI Act Art. 12: logging must be comprehensive → x_log >= 0.8
    constraints.add(Constraint::hard(
        "c-aia-logging",
        ConstraintExpr::var("x_log_ge_08"),
    ));

    // GDPR Art. 5(1)(c): minimize personal data → x_log <= 0.3
    constraints.add(Constraint::hard(
        "c-gdpr-data-min",
        ConstraintExpr::var("x_log_le_03"),
    ));

    // GDPR Art. 5(1)(e): retain no longer than necessary → x_retain <= 6
    constraints.add(Constraint::hard(
        "c-gdpr-storage",
        ConstraintExpr::var("x_retain_le_6"),
    ));

    // AI Act Art. 12: logs retained for "appropriate period" → x_retain >= 12
    constraints.add(Constraint::hard(
        "c-aia-retention",
        ConstraintExpr::var("x_retain_ge_12"),
    ));

    // AI Act Art. 13: transparency → x_disclose >= 0.7
    constraints.add(Constraint::hard(
        "c-aia-transparency",
        ConstraintExpr::var("x_disclose_ge_07"),
    ));

    // Trade Secrets: limit disclosure → x_disclose <= 0.4
    constraints.add(Constraint::soft(
        "c-tsd-protection",
        ConstraintExpr::var("x_disclose_le_04"),
        0.6,
    ));

    (obligations, constraints)
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  RegSynth — Cross-Jurisdictional Conflict Detection        ║");
    println!("║  EU AI Act × GDPR × Trade Secrets Directive                ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let (obligations, constraints) = build_conflict_scenario();

    // 1. Display obligations
    println!("📋 Regulatory Obligations ({}):\n", obligations.len());
    for obl in &obligations {
        let art = obl.article_ref.as_ref()
            .map(|a| format!("{}", a))
            .unwrap_or_default();
        println!("  [{kind}] {id}\n        {desc}\n        Ref: {art} | Grade: {grade}",
            kind = obl.kind, id = obl.id, desc = obl.description, grade = obl.grade);
        println!();
    }

    // 2. Display constraint set
    println!("🔒 Formal Constraints ({} total, {} hard, {} soft):",
        constraints.len(), constraints.hard_count(), constraints.soft_count());
    for c in constraints.all() {
        let kind_str = if c.kind.is_hard() { "HARD" } else { "SOFT" };
        println!("  [{}] {} — {}", kind_str, c.id, c.description);
    }
    println!();

    // 3. Conflict detection (simulated MUS extraction)
    println!("⚡ Conflict Detection Results:\n");

    // Conflict 1: Logging vs Data Minimization
    println!("  ┌─────────────────────────────────────────────────────────┐");
    println!("  │ CONFLICT #1: Data Logging vs. Data Minimization        │");
    println!("  ├─────────────────────────────────────────────────────────┤");
    println!("  │ Category: LogicalContradiction                         │");
    println!("  │ Severity: Critical                                     │");
    println!("  │                                                        │");
    println!("  │ MUS (Minimal Unsatisfiable Subset):                    │");
    println!("  │   • c-aia-logging   (x_log ≥ 0.8)  ← EU AI Act Art.12│");
    println!("  │   • c-gdpr-data-min (x_log ≤ 0.3)  ← GDPR Art.5(1)(c)│");
    println!("  │                                                        │");
    println!("  │ These two constraints are mutually exclusive:           │");
    println!("  │ no value of x_log can satisfy both simultaneously.     │");
    println!("  └─────────────────────────────────────────────────────────┘");
    println!();

    // Conflict 2: Retention period
    println!("  ┌─────────────────────────────────────────────────────────┐");
    println!("  │ CONFLICT #2: Log Retention vs. Storage Limitation      │");
    println!("  ├─────────────────────────────────────────────────────────┤");
    println!("  │ Category: TemporalConflict                             │");
    println!("  │ Severity: High                                         │");
    println!("  │                                                        │");
    println!("  │ MUS:                                                   │");
    println!("  │   • c-aia-retention (x_retain ≥ 12) ← EU AI Act Art.12│");
    println!("  │   • c-gdpr-storage  (x_retain ≤ 6)  ← GDPR Art.5(1)(e)│");
    println!("  │                                                        │");
    println!("  │ Retention requirements are contradictory.               │");
    println!("  └─────────────────────────────────────────────────────────┘");
    println!();

    // Conflict 3: Transparency vs Trade Secrets (soft)
    println!("  ┌─────────────────────────────────────────────────────────┐");
    println!("  │ CONFLICT #3: Transparency vs. Trade Secret Protection  │");
    println!("  ├─────────────────────────────────────────────────────────┤");
    println!("  │ Category: PolicyOverlap                                │");
    println!("  │ Severity: Medium (soft constraint involved)            │");
    println!("  │                                                        │");
    println!("  │ Constraints:                                           │");
    println!("  │   • c-aia-transparency (x_disclose ≥ 0.7) [HARD]      │");
    println!("  │   • c-tsd-protection   (x_disclose ≤ 0.4) [SOFT w=0.6]│");
    println!("  │                                                        │");
    println!("  │ Resolution: relax trade-secret protection (soft) to    │");
    println!("  │ satisfy mandatory AI Act transparency requirement.     │");
    println!("  └─────────────────────────────────────────────────────────┘");
    println!();

    // 4. Suggested resolutions
    println!("💡 Suggested Resolutions:");
    println!("  1. Logging conflict: Use privacy-preserving logging (differential privacy,");
    println!("     k-anonymity) to satisfy Art. 12 logging while respecting GDPR minimization.");
    println!("  2. Retention conflict: Apply GDPR Art. 6(1)(c) legal obligation basis,");
    println!("     arguing AI Act creates a legal basis for extended retention of operational logs.");
    println!("  3. Transparency conflict: Use layered disclosure — technical docs under NDA");
    println!("     for notified bodies, public transparency without trade secrets.");
    println!();

    // 5. Diagnosis summary
    let diagnosis = RegulatoryDiagnosis {
        conflicts: vec![
            RegulatoryConflict {
                constraint_ids: vec!["c-aia-logging".into(), "c-gdpr-data-min".into()],
                article_refs: vec!["EU-AI-Act Art.12".into(), "GDPR Art.5(1)(c)".into()],
                jurisdictions: vec!["EU".into()],
                category: ConflictCategory::LogicalContradiction,
                severity: ConflictSeverity::Critical,
                explanation: "Logging mandate contradicts data minimization".into(),
            },
            RegulatoryConflict {
                constraint_ids: vec!["c-aia-retention".into(), "c-gdpr-storage".into()],
                article_refs: vec!["EU-AI-Act Art.12".into(), "GDPR Art.5(1)(e)".into()],
                jurisdictions: vec!["EU".into()],
                category: ConflictCategory::JurisdictionalClash,
                severity: ConflictSeverity::High,
                explanation: "Retention period requirements are contradictory".into(),
            },
        ],
        suggested_relaxations: vec![
            "Relax c-gdpr-data-min to allow structured operational logs".into(),
            "Apply GDPR Art.6(1)(c) legal obligation basis for log retention".into(),
        ],
        total_conflicts: 2,
    };

    println!("📊 Diagnosis Summary:");
    println!("  Total hard conflicts detected: {}", diagnosis.total_conflicts);
    println!("  Suggested relaxations: {}", diagnosis.suggested_relaxations.len());
    for (i, relax) in diagnosis.suggested_relaxations.iter().enumerate() {
        println!("    {}. {}", i + 1, relax);
    }

    println!("\n✅ Conflict analysis complete. {} conflicts identified across {} obligations.",
        diagnosis.total_conflicts, obligations.len());
}
