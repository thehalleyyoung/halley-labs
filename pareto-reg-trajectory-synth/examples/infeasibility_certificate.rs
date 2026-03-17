//! # Infeasibility Certificate Generation with MUS Extraction
//!
//! This example demonstrates generating a formal infeasibility certificate
//! when a set of regulatory obligations cannot all be simultaneously satisfied.
//!
//! ## Scenario: Conflicting Multi-Jurisdictional Requirements
//!
//! A company deploys an AI system that must comply with:
//!
//! 1. **EU AI Act Art. 10(2)(f)**: Training data must include data from
//!    persons belonging to groups likely to be affected (diversity requirement).
//!
//! 2. **GDPR Art. 9(1)**: Processing of special category data (racial origin,
//!    political opinions, health data, etc.) is prohibited without explicit
//!    consent or legal basis.
//!
//! 3. **EU AI Act Art. 10(5)**: Training data for bias detection may include
//!    special-category data, but only to the extent strictly necessary.
//!
//! 4. **California CCPA §1798.140(o)**: Personal information includes
//!    biometric and demographic data; opt-out rights apply.
//!
//! These constraints form a satisfiability problem. When the problem is UNSAT,
//! MUS (Minimal Unsatisfiable Subset) extraction identifies the minimal set
//! of conflicting constraints, enabling targeted regulatory diagnosis.
//!
//! ## Certificate Structure
//!
//! The infeasibility certificate contains:
//! - Resolution proof (sequence of resolution steps deriving ⊥)
//! - MUS mapping to regulatory articles
//! - Conflict categorization and severity
//! - SHA-256 fingerprint for tamper detection

use regsynth_types::{
    ArticleRef, FormalizabilityGrade, Jurisdiction, ObligationKind, RiskLevel,
    TemporalInterval,
    constraint::{
        Constraint, ConstraintExpr, ConstraintKind, ConstraintSet, VarId,
    },
    certificate::{
        Certificate, CertificateKind, ProofWitness, ResolutionProof, ResolutionStep,
    },
};
use regsynth_temporal::Obligation;
use regsynth_encoding::{SmtExpr, SmtSort, SmtConstraint, Provenance};
use regsynth_certificate::{
    InfeasibilityCertGenerator, InfeasibilityCertificate,
    MinimalUnsatisfiableSubset, MusConstraint,
    RegulatoryConflict, RegulatoryDiagnosis,
    ConflictCategory, ConflictSeverity,
};
use regsynth_solver::{ConflictCore, ConflictType};

/// Build the regulatory obligations for this scenario.
fn build_obligations() -> Vec<Obligation> {
    let eu = Jurisdiction::new("EU");
    let ca = Jurisdiction::new("US-CA");

    vec![
        Obligation::new(
            "eu-aia-art10-diversity",
            ObligationKind::Obligation,
            eu.clone(),
            "Training data must include data representative of affected groups (Art. 10(2)(f))",
        )
        .with_article_ref(ArticleRef {
            framework: "EU-AI-Act".into(),
            article: "10".into(),
            paragraph: Some("2(f)".into()),
        })
        .with_risk_level(RiskLevel::High)
        .with_grade(FormalizabilityGrade::F3),

        Obligation::new(
            "gdpr-art9-special-cat",
            ObligationKind::Prohibition,
            eu.clone(),
            "Processing of special-category data prohibited without legal basis (Art. 9(1))",
        )
        .with_article_ref(ArticleRef {
            framework: "GDPR".into(),
            article: "9".into(),
            paragraph: Some("1".into()),
        })
        .with_risk_level(RiskLevel::High)
        .with_grade(FormalizabilityGrade::F2),

        Obligation::new(
            "eu-aia-art10-bias-data",
            ObligationKind::Permission,
            eu.clone(),
            "May process special-category data for bias detection, strictly necessary (Art. 10(5))",
        )
        .with_article_ref(ArticleRef {
            framework: "EU-AI-Act".into(),
            article: "10".into(),
            paragraph: Some("5".into()),
        })
        .with_risk_level(RiskLevel::High)
        .with_grade(FormalizabilityGrade::F4),

        Obligation::new(
            "ccpa-optout",
            ObligationKind::Obligation,
            ca.clone(),
            "Honor consumer opt-out requests for personal data processing (CCPA §1798.120)",
        )
        .with_article_ref(ArticleRef {
            framework: "CCPA".into(),
            article: "1798.120".into(),
            paragraph: None,
        })
        .with_risk_level(RiskLevel::High)
        .with_grade(FormalizabilityGrade::F2),
    ]
}

/// Build the formal constraint set.
fn build_constraints() -> ConstraintSet {
    let mut cs = ConstraintSet::new();

    // AI Act Art. 10(2)(f): diversity data must be collected → x_diversity_data ≥ 0.7
    let mut c1 = Constraint::hard("c-aia-diversity", ConstraintExpr::var("x_diversity_ge_07"));
    c1.description = "Training data diversity ≥ 0.7".into();
    c1.source_obligation = Some("eu-aia-art10-diversity".into());
    c1.source_jurisdiction = Some("EU".into());
    cs.add(c1);

    // GDPR Art. 9(1): special-category processing prohibited → x_special_cat = 0
    let mut c2 = Constraint::hard("c-gdpr-special-cat", ConstraintExpr::var("x_special_cat_eq_0"));
    c2.description = "Special-category data processing = 0 (without legal basis)".into();
    c2.source_obligation = Some("gdpr-art9-special-cat".into());
    c2.source_jurisdiction = Some("EU".into());
    cs.add(c2);

    // AI Act Art. 10(5): bias data allowed but limited → x_special_cat ≤ 0.3
    let mut c3 = Constraint::soft("c-aia-bias-exception", ConstraintExpr::var("x_special_cat_le_03"), 0.7);
    c3.description = "Special-category data for bias ≤ 0.3".into();
    c3.source_obligation = Some("eu-aia-art10-bias-data".into());
    c3.source_jurisdiction = Some("EU".into());
    cs.add(c3);

    // CCPA: opt-out honored → x_data_retained_after_optout = 0
    let mut c4 = Constraint::hard("c-ccpa-optout", ConstraintExpr::var("x_no_data_after_optout"));
    c4.description = "No personal data retained post opt-out".into();
    c4.source_obligation = Some("ccpa-optout".into());
    c4.source_jurisdiction = Some("US-CA".into());
    cs.add(c4);

    // AI Act Art. 10(2)(f) implies special-category data needed:
    // diversity ≥ 0.7 → special_cat > 0 (to achieve diversity, need demographic data)
    let mut c5 = Constraint::hard("c-diversity-implies-special",
        ConstraintExpr::var("x_diversity_implies_special_cat"));
    c5.description = "Achieving diversity requires some special-category data".into();
    cs.add(c5);

    cs
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  RegSynth — Infeasibility Certificate Generation           ║");
    println!("║  MUS Extraction for Regulatory Conflicts                   ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let obligations = build_obligations();
    let constraints = build_constraints();

    // 1. Display obligations
    println!("📋 Regulatory Obligations:\n");
    for obl in &obligations {
        let art = obl.article_ref.as_ref().map(|a| format!("{}", a)).unwrap_or_default();
        println!("  [{kind}] {id}\n        {desc}\n        Ref: {art} | Grade: {grade}\n",
            kind = obl.kind, id = obl.id, desc = obl.description, grade = obl.grade);
    }

    // 2. Display constraints
    println!("🔒 Constraints ({} total, {} hard, {} soft):\n",
        constraints.len(), constraints.hard_count(), constraints.soft_count());
    for c in constraints.all() {
        let kind_str = if c.kind.is_hard() { "HARD" } else {
            &format!("SOFT(w={:.1})", c.kind.weight())
        };
        println!("  [{}] {} — {}", kind_str, c.id, c.description);
    }
    println!();

    // 3. Simulated MUS extraction result
    let mus = MinimalUnsatisfiableSubset {
        constraints: vec![
            MusConstraint {
                constraint_id: "c-aia-diversity".into(),
                description: "Training data diversity ≥ 0.7".into(),
                provenance: Some(Provenance {
                    obligation_id: "eu-aia-art10-diversity".into(),
                    jurisdiction: "EU".into(),
                    article_ref: Some("EU-AI-Act Art. 10(2)(f)".into()),
                    description: "Diversity requirement for training data".into(),
                }),
            },
            MusConstraint {
                constraint_id: "c-gdpr-special-cat".into(),
                description: "Special-category data processing = 0".into(),
                provenance: Some(Provenance {
                    obligation_id: "gdpr-art9-special-cat".into(),
                    jurisdiction: "EU".into(),
                    article_ref: Some("GDPR Art. 9(1)".into()),
                    description: "Prohibition on special-category data".into(),
                }),
            },
            MusConstraint {
                constraint_id: "c-diversity-implies-special".into(),
                description: "Diversity requires some special-category data".into(),
                provenance: None,
            },
        ],
        size: 3,
    };

    println!("⚡ MUS Extraction Result:\n");
    println!("  Minimal Unsatisfiable Subset (size = {}):", mus.size);
    for mc in &mus.constraints {
        let prov = mc.provenance.as_ref()
            .map(|p| format!("← {} ({})", p.article_ref.as_deref().unwrap_or("N/A"), p.jurisdiction))
            .unwrap_or_else(|| "← derived".into());
        println!("    • {} — {} {}", mc.constraint_id, mc.description, prov);
    }
    println!();

    // 4. Conflict diagnosis
    let conflict_core = ConflictCore::new(
        vec![regsynth_types::Id::new(), regsynth_types::Id::new()],
        "Training data diversity requirement (Art. 10(2)(f)) \
         requires special-category personal data (demographic info), \
         but GDPR Art. 9(1) prohibits processing such data without \
         explicit consent or legal basis.",
        ConflictType::LogicalContradiction,
    );

    println!("🔍 Conflict Core Analysis:");
    println!("  Type: {:?}", conflict_core.conflict_type);
    println!("  Size: {} obligations", conflict_core.size());
    println!("  Explanation: {}", conflict_core.explanation);
    println!();

    // 5. Build resolution proof
    let proof = ResolutionProof::new();
    let cert = Certificate::new(
        CertificateKind::Infeasibility,
        "RegSynth-MaxSMT-v0.1",
        ProofWitness::ResolutionProof(proof),
    );

    println!("📜 Infeasibility Certificate:");
    println!("  ID:          {}", cert.id);
    println!("  Kind:        {}", cert.kind);
    println!("  Solver:      {}", cert.solver_used);
    println!("  Timestamp:   {}", cert.timestamp);
    println!("  Fingerprint: {}", &cert.fingerprint[..32]);
    println!("  Integrity:   {}", if cert.verify_integrity() { "✅ VALID" } else { "❌ TAMPERED" });
    println!();

    // 6. Suggested relaxations
    println!("💡 Suggested Relaxations:");
    println!("  1. Invoke GDPR Art. 9(2)(g): processing necessary for substantial");
    println!("     public interest, with basis in EU law (the AI Act itself).");
    println!("  2. Apply AI Act Art. 10(5) exception: use special-category data");
    println!("     solely for bias detection under strict safeguards.");
    println!("  3. Obtain explicit consent (GDPR Art. 9(2)(a)) from data subjects");
    println!("     for inclusion of demographic data in training sets.");
    println!("  4. Use synthetic demographic data to satisfy diversity without");
    println!("     processing real special-category data.");

    println!("\n✅ Infeasibility certificate generated with {} MUS constraints.", mus.size);
}
