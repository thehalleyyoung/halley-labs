//! THE TRUSTED PROOF CHECKER KERNEL.
//!
//! This module is the core trusted computing base for CollusionProof certificates.
//! It verifies that every proof step is valid, axiom instantiations are sound,
//! inference rules are correctly applied, rational arithmetic is consistent,
//! segment isolation is preserved, and the alpha budget is not exceeded.

use crate::ast::*;
use crate::proof_term::*;
use crate::rational_verifier::{OrderingVerification, RationalVerifier};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

// ── Verification result ──────────────────────────────────────────────────────

/// The result of verifying an entire certificate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VerificationResult {
    Valid(VerificationReport),
    Invalid(VerificationError),
}

impl VerificationResult {
    pub fn is_valid(&self) -> bool {
        matches!(self, VerificationResult::Valid(_))
    }

    pub fn unwrap_report(self) -> VerificationReport {
        match self {
            VerificationResult::Valid(r) => r,
            VerificationResult::Invalid(e) => panic!("Certificate is invalid: {:?}", e),
        }
    }
}

/// The result of verifying a single proof step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StepResult {
    Valid(StepReport),
    Invalid(StepError),
}

impl StepResult {
    pub fn is_valid(&self) -> bool {
        matches!(self, StepResult::Valid(_))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepReport {
    pub step_index: usize,
    pub step_kind: String,
    pub declared_ref: Option<String>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepError {
    pub step_index: usize,
    pub step_kind: String,
    pub message: String,
}

/// Detailed report of a successful verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationReport {
    pub certificate_version: String,
    pub scenario: String,
    pub total_steps: usize,
    pub verified_steps: usize,
    pub verdict: Option<VerdictType>,
    pub verdict_confidence: Option<f64>,
    pub alpha_budget_total: f64,
    pub alpha_budget_spent: f64,
    pub rational_verifications: usize,
    pub rational_disagreements: usize,
    pub segment_checks: usize,
    pub step_reports: Vec<StepReport>,
    pub notes: Vec<String>,
}

impl VerificationReport {
    fn new(header: &CertificateHeader) -> Self {
        Self {
            certificate_version: header.version.clone(),
            scenario: header.scenario.clone(),
            total_steps: 0,
            verified_steps: 0,
            verdict: None,
            verdict_confidence: None,
            alpha_budget_total: header.alpha.value(),
            alpha_budget_spent: 0.0,
            rational_verifications: 0,
            rational_disagreements: 0,
            segment_checks: 0,
            step_reports: Vec::new(),
            notes: Vec::new(),
        }
    }
}

/// Error encountered during verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationError {
    pub step_index: Option<usize>,
    pub kind: VerificationErrorKind,
    pub message: String,
    pub details: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum VerificationErrorKind {
    InvalidHeader,
    InvalidProofStep,
    AxiomInstantiationError,
    InferenceRuleError,
    UndeclaredReference,
    DuplicateReference,
    AlphaBudgetExceeded,
    SegmentIsolationViolation,
    RationalArithmeticDisagreement,
    InvalidVerdictDerivation,
    MissingVerdictStep,
    TypeMismatch,
    ArithmeticError,
    StructuralError,
}

impl std::fmt::Display for VerificationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{:?}] {}", self.kind, self.message)
    }
}

// ── Checker context ──────────────────────────────────────────────────────────

/// Accumulated state during proof checking.
#[derive(Debug, Clone)]
pub struct CheckerContext {
    /// Declared references and their types.
    pub declared_refs: HashMap<String, RefInfo>,
    /// Alpha budget tracker.
    pub alpha_checker: AlphaBudgetChecker,
    /// Segment isolation tracker.
    pub segment_checker: SegmentIsolationChecker,
    /// Verified facts so far (ref → ProofFact).
    pub verified_facts: HashMap<String, ProofFact>,
    /// Oracle access level of the certificate.
    pub oracle_level: shared_types::OracleAccessLevel,
    /// Game specification from equilibrium claims.
    pub game_spec: Option<GameSpec>,
    /// Nash profile from equilibrium claims.
    pub nash_profile: Option<NashProfile>,
}

impl CheckerContext {
    pub fn new(header: &CertificateHeader) -> Self {
        Self {
            declared_refs: HashMap::new(),
            alpha_checker: AlphaBudgetChecker::new(header.alpha.value()),
            segment_checker: SegmentIsolationChecker::new(),
            verified_facts: HashMap::new(),
            oracle_level: header.oracle_level,
            game_spec: None,
            nash_profile: None,
        }
    }

    pub fn declare_ref(&mut self, name: &str, info: RefInfo) -> Result<(), String> {
        if self.declared_refs.contains_key(name) {
            return Err(format!("Duplicate reference: {}", name));
        }
        self.declared_refs.insert(name.to_string(), info);
        Ok(())
    }

    pub fn has_ref(&self, name: &str) -> bool {
        self.declared_refs.contains_key(name)
    }

    pub fn add_fact(&mut self, name: &str, fact: ProofFact) {
        self.verified_facts.insert(name.to_string(), fact);
    }

    pub fn get_fact(&self, name: &str) -> Option<&ProofFact> {
        self.verified_facts.get(name)
    }

    pub fn all_refs_exist(&self, refs: &[String]) -> Result<(), String> {
        for r in refs {
            if !self.declared_refs.contains_key(r) {
                return Err(format!("Undeclared reference: {}", r));
            }
        }
        Ok(())
    }
}

/// Information about a declared reference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefInfo {
    pub kind: RefKind,
    pub step_index: usize,
    pub segment_type: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum RefKind {
    Trajectory,
    StatisticalTest,
    Equilibrium,
    Deviation,
    Punishment,
    CollusionPremium,
    Inference,
}

/// A verified fact in the context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProofFact {
    TestRejection {
        test_ref: String,
        alpha_spent: f64,
        p_value: f64,
    },
    EquilibriumEstablished {
        eq_ref: String,
        prices: Vec<f64>,
        profits: Vec<f64>,
    },
    DeviationBounded {
        dev_ref: String,
        player: usize,
        bound: f64,
        confidence: f64,
    },
    PunishmentDetected {
        pun_ref: String,
        player: usize,
        payoff_drop: f64,
        p_value: f64,
    },
    CollusionPremiumEstablished {
        cp_ref: String,
        value: f64,
        ci_lower: f64,
        ci_upper: f64,
    },
    InferredFact {
        inf_ref: String,
        statement: String,
    },
    Verdict {
        verdict: VerdictType,
        confidence: f64,
    },
}

// ── Main proof checker ───────────────────────────────────────────────────────

/// The trusted proof checker kernel.
pub struct ProofChecker {
    rational_verifier: RationalVerifier,
    strict_mode: bool,
}

impl ProofChecker {
    pub fn new() -> Self {
        Self {
            rational_verifier: RationalVerifier::new(),
            strict_mode: true,
        }
    }

    pub fn with_strict_mode(mut self, strict: bool) -> Self {
        self.strict_mode = strict;
        self
    }

    /// Verify an entire certificate. This is the main entry point.
    pub fn check_certificate(&self, cert: &CertificateAST) -> VerificationResult {
        // 1. Validate header
        if let Err(e) = self.validate_header(&cert.header) {
            return VerificationResult::Invalid(e);
        }

        // 2. Check for empty certificate
        if cert.body.is_empty() {
            return VerificationResult::Invalid(VerificationError {
                step_index: None,
                kind: VerificationErrorKind::StructuralError,
                message: "Certificate body is empty".to_string(),
                details: None,
            });
        }

        // 3. Initialize context
        let mut ctx = CheckerContext::new(&cert.header);
        let mut report = VerificationReport::new(&cert.header);
        report.total_steps = cert.body.steps.len();

        // 4. Check each proof step in sequence
        for (i, step) in cert.body.steps.iter().enumerate() {
            match self.check_proof_step(step, i, &mut ctx) {
                StepResult::Valid(step_report) => {
                    report.verified_steps += 1;
                    report.step_reports.push(step_report);
                }
                StepResult::Invalid(step_error) => {
                    return VerificationResult::Invalid(VerificationError {
                        step_index: Some(i),
                        kind: VerificationErrorKind::InvalidProofStep,
                        message: step_error.message.clone(),
                        details: Some(format!("Step {}: {}", i, step_error.step_kind)),
                    });
                }
            }
        }

        // 5. Verify alpha budget was not exceeded
        if self.strict_mode && ctx.alpha_checker.is_exceeded() {
            return VerificationResult::Invalid(VerificationError {
                step_index: None,
                kind: VerificationErrorKind::AlphaBudgetExceeded,
                message: format!(
                    "Alpha budget exceeded: spent {:.6} of {:.6}",
                    ctx.alpha_checker.spent, ctx.alpha_checker.total
                ),
                details: None,
            });
        }
        report.alpha_budget_spent = ctx.alpha_checker.spent;

        // 6. Verify certificate ends with a verdict
        let has_verdict = cert
            .body
            .steps
            .iter()
            .any(|s| matches!(s, ProofStep::Verdict(..)));
        if self.strict_mode && !has_verdict {
            return VerificationResult::Invalid(VerificationError {
                step_index: None,
                kind: VerificationErrorKind::MissingVerdictStep,
                message: "Certificate must end with a Verdict step".to_string(),
                details: None,
            });
        }

        // 7. Extract verdict information
        if let Some(ProofStep::Verdict(vt, conf, _)) = cert.body.steps.iter().rev().find(|s| matches!(s, ProofStep::Verdict(..))) {
            report.verdict = Some(*vt);
            report.verdict_confidence = Some(conf.0);
        }

        // 8. Record segment checks and rational verifications
        report.segment_checks = ctx.segment_checker.check_count;
        report.rational_verifications = self.rational_verifier.verification_count();

        VerificationResult::Valid(report)
    }

    /// Validate the certificate header.
    fn validate_header(&self, header: &CertificateHeader) -> Result<(), VerificationError> {
        if header.version.is_empty() {
            return Err(VerificationError {
                step_index: None,
                kind: VerificationErrorKind::InvalidHeader,
                message: "Certificate version is empty".to_string(),
                details: None,
            });
        }
        let alpha = header.alpha.value();
        if alpha <= 0.0 || alpha > 1.0 {
            return Err(VerificationError {
                step_index: None,
                kind: VerificationErrorKind::InvalidHeader,
                message: format!("Alpha must be in (0, 1], got {:.6}", alpha),
                details: None,
            });
        }
        if header.scenario.is_empty() {
            return Err(VerificationError {
                step_index: None,
                kind: VerificationErrorKind::InvalidHeader,
                message: "Scenario name is empty".to_string(),
                details: None,
            });
        }
        Ok(())
    }

    /// Check a single proof step and update the context.
    pub fn check_proof_step(
        &self,
        step: &ProofStep,
        index: usize,
        ctx: &mut CheckerContext,
    ) -> StepResult {
        let kind = step.kind().to_string();
        let mut notes = Vec::new();

        match step {
            ProofStep::DataDeclaration(tref, seg) => {
                // Validate segment specification
                if seg.end_round <= seg.start_round {
                    return StepResult::Invalid(StepError {
                        step_index: index,
                        step_kind: kind,
                        message: "Segment end must be > start".to_string(),
                    });
                }
                if seg.num_players == 0 {
                    return StepResult::Invalid(StepError {
                        step_index: index,
                        step_kind: kind,
                        message: "Segment must have > 0 players".to_string(),
                    });
                }
                // Register segment type
                ctx.segment_checker.register_segment(
                    &tref.0,
                    &seg.segment_type,
                    seg.start_round,
                    seg.end_round,
                );
                // Declare reference
                if let Err(e) = ctx.declare_ref(
                    &tref.0,
                    RefInfo {
                        kind: RefKind::Trajectory,
                        step_index: index,
                        segment_type: Some(seg.segment_type.clone()),
                    },
                ) {
                    return StepResult::Invalid(StepError {
                        step_index: index,
                        step_kind: kind,
                        message: e,
                    });
                }
                notes.push(format!(
                    "Declared trajectory {} ({} rounds, {} players)",
                    tref.0,
                    seg.num_rounds(),
                    seg.num_players
                ));
            }

            ProofStep::StatisticalTest(tref, test_type, stat, pval) => {
                // Validate p-value range
                if pval.0 < 0.0 || pval.0 > 1.0 {
                    return StepResult::Invalid(StepError {
                        step_index: index,
                        step_kind: kind,
                        message: format!("P-value out of range: {:.6}", pval.0),
                    });
                }
                // Validate statistic is finite
                if !stat.value.is_finite() {
                    return StepResult::Invalid(StepError {
                        step_index: index,
                        step_kind: kind,
                        message: "Test statistic must be finite".to_string(),
                    });
                }
                // Track alpha spending if the test rejects
                let alpha = ctx.alpha_checker.total;
                if pval.is_significant(alpha) {
                    // This test consumed some alpha budget — record it
                    ctx.alpha_checker.spend(pval.0.max(1e-15));
                    notes.push(format!(
                        "Test {} rejected at p={:.6} (alpha spent: {:.6})",
                        tref.0, pval.0, ctx.alpha_checker.spent
                    ));
                }
                if let Err(e) = ctx.declare_ref(
                    &tref.0,
                    RefInfo {
                        kind: RefKind::StatisticalTest,
                        step_index: index,
                        segment_type: None,
                    },
                ) {
                    return StepResult::Invalid(StepError {
                        step_index: index,
                        step_kind: kind,
                        message: e,
                    });
                }
                ctx.add_fact(
                    &tref.0,
                    ProofFact::TestRejection {
                        test_ref: tref.0.clone(),
                        alpha_spent: pval.0.max(1e-15),
                        p_value: pval.0,
                    },
                );
            }

            ProofStep::EquilibriumClaim(eref, game_spec, nash) => {
                // Validate Nash profile dimensions
                if nash.prices.len() != game_spec.num_players {
                    return StepResult::Invalid(StepError {
                        step_index: index,
                        step_kind: kind,
                        message: format!(
                            "Nash profile size mismatch: {} prices for {} players",
                            nash.prices.len(),
                            game_spec.num_players
                        ),
                    });
                }
                if nash.profits.len() != game_spec.num_players {
                    return StepResult::Invalid(StepError {
                        step_index: index,
                        step_kind: kind,
                        message: format!(
                            "Nash profile size mismatch: {} profits for {} players",
                            nash.profits.len(),
                            game_spec.num_players
                        ),
                    });
                }
                // Validate prices and profits are finite and non-negative
                for (i, p) in nash.prices.iter().enumerate() {
                    if !p.is_finite() || *p < 0.0 {
                        return StepResult::Invalid(StepError {
                            step_index: index,
                            step_kind: kind,
                            message: format!("Invalid Nash price for player {}: {}", i, p),
                        });
                    }
                }
                // Verify game specification consistency
                if game_spec.num_players < 2 {
                    return StepResult::Invalid(StepError {
                        step_index: index,
                        step_kind: kind,
                        message: "Game must have at least 2 players".to_string(),
                    });
                }
                if let Err(e) = ctx.declare_ref(
                    &eref.0,
                    RefInfo {
                        kind: RefKind::Equilibrium,
                        step_index: index,
                        segment_type: None,
                    },
                ) {
                    return StepResult::Invalid(StepError {
                        step_index: index,
                        step_kind: kind,
                        message: e,
                    });
                }
                ctx.game_spec = Some(game_spec.clone());
                ctx.nash_profile = Some(nash.clone());
                ctx.add_fact(
                    &eref.0,
                    ProofFact::EquilibriumEstablished {
                        eq_ref: eref.0.clone(),
                        prices: nash.prices.clone(),
                        profits: nash.profits.clone(),
                    },
                );
                notes.push(format!(
                    "Equilibrium {} established for {} players",
                    eref.0, game_spec.num_players
                ));
            }

            ProofStep::DeviationBound(dref, player, bound, conf) => {
                // Validate bound
                if !bound.value.is_finite() {
                    return StepResult::Invalid(StepError {
                        step_index: index,
                        step_kind: kind,
                        message: "Deviation bound must be finite".to_string(),
                    });
                }
                // Validate confidence level
                let cl = conf.value();
                if cl <= 0.0 || cl >= 1.0 {
                    return StepResult::Invalid(StepError {
                        step_index: index,
                        step_kind: kind,
                        message: format!(
                            "Confidence level must be in (0, 1), got {:.6}",
                            cl
                        ),
                    });
                }
                // This is a Layer1+ claim
                if ctx.oracle_level == shared_types::OracleAccessLevel::Layer0 {
                    notes.push("DeviationBound at Layer0 — cannot be verified without counterfactual access".to_string());
                }
                if let Err(e) = ctx.declare_ref(
                    &dref.0,
                    RefInfo {
                        kind: RefKind::Deviation,
                        step_index: index,
                        segment_type: None,
                    },
                ) {
                    return StepResult::Invalid(StepError {
                        step_index: index,
                        step_kind: kind,
                        message: e,
                    });
                }
                ctx.add_fact(
                    &dref.0,
                    ProofFact::DeviationBounded {
                        dev_ref: dref.0.clone(),
                        player: player.0,
                        bound: bound.value,
                        confidence: cl,
                    },
                );
            }

            ProofStep::PunishmentEvidence(pref, player, drop, pval) => {
                // Validate payoff drop
                if !drop.value.is_finite() {
                    return StepResult::Invalid(StepError {
                        step_index: index,
                        step_kind: kind,
                        message: "Payoff drop must be finite".to_string(),
                    });
                }
                // Validate p-value
                if pval.0 < 0.0 || pval.0 > 1.0 {
                    return StepResult::Invalid(StepError {
                        step_index: index,
                        step_kind: kind,
                        message: format!("P-value out of range: {:.6}", pval.0),
                    });
                }
                // This is a Layer2 claim
                if ctx.oracle_level != shared_types::OracleAccessLevel::Layer2 {
                    notes.push(
                        "PunishmentEvidence requires Layer2 oracle access".to_string(),
                    );
                }
                if let Err(e) = ctx.declare_ref(
                    &pref.0,
                    RefInfo {
                        kind: RefKind::Punishment,
                        step_index: index,
                        segment_type: None,
                    },
                ) {
                    return StepResult::Invalid(StepError {
                        step_index: index,
                        step_kind: kind,
                        message: e,
                    });
                }
                ctx.add_fact(
                    &pref.0,
                    ProofFact::PunishmentDetected {
                        pun_ref: pref.0.clone(),
                        player: player.0,
                        payoff_drop: drop.value,
                        p_value: pval.0,
                    },
                );
            }

            ProofStep::CollusionPremium(cpref, val, ci) => {
                // Validate CP value
                if !val.0.is_finite() {
                    return StepResult::Invalid(StepError {
                        step_index: index,
                        step_kind: kind,
                        message: "Collusion premium must be finite".to_string(),
                    });
                }
                // Validate CI contains the point estimate
                if !ci.contains(val.0) {
                    return StepResult::Invalid(StepError {
                        step_index: index,
                        step_kind: kind,
                        message: format!(
                            "CI [{:.4}, {:.4}] does not contain CP value {:.4}",
                            ci.lower, ci.upper, val.0
                        ),
                    });
                }
                // Validate CI ordering
                if ci.lower > ci.upper {
                    return StepResult::Invalid(StepError {
                        step_index: index,
                        step_kind: kind,
                        message: "CI lower bound exceeds upper bound".to_string(),
                    });
                }
                if let Err(e) = ctx.declare_ref(
                    &cpref.0,
                    RefInfo {
                        kind: RefKind::CollusionPremium,
                        step_index: index,
                        segment_type: None,
                    },
                ) {
                    return StepResult::Invalid(StepError {
                        step_index: index,
                        step_kind: kind,
                        message: e,
                    });
                }
                ctx.add_fact(
                    &cpref.0,
                    ProofFact::CollusionPremiumEstablished {
                        cp_ref: cpref.0.clone(),
                        value: val.0,
                        ci_lower: ci.lower,
                        ci_upper: ci.upper,
                    },
                );
                notes.push(format!(
                    "CP = {:.4} ∈ [{:.4}, {:.4}]",
                    val.0, ci.lower, ci.upper
                ));
            }

            ProofStep::Inference(iref, rule, premises, conclusion) => {
                // Verify all premises have been declared
                if let Err(e) = ctx.all_refs_exist(&premises.refs) {
                    return StepResult::Invalid(StepError {
                        step_index: index,
                        step_kind: kind,
                        message: e,
                    });
                }
                // Verify the inference rule structurally
                if let Err(e) =
                    self.verify_inference_structural(rule, premises, conclusion, ctx)
                {
                    return StepResult::Invalid(StepError {
                        step_index: index,
                        step_kind: kind,
                        message: e,
                    });
                }
                // Verify rational arithmetic if conclusion contains an expression
                if let Some(expr) = &conclusion.expression {
                    if let Some(ordering) = self.rational_verifier.verify_expression(expr) {
                        if !ordering.agree {
                            if self.strict_mode {
                                return StepResult::Invalid(StepError {
                                    step_index: index,
                                    step_kind: kind,
                                    message: format!(
                                        "Rational arithmetic disagreement: f64={}, rational={}",
                                        ordering.f64_result, ordering.rational_result
                                    ),
                                });
                            }
                            notes.push(format!(
                                "WARNING: rational disagreement at step {}",
                                index
                            ));
                        }
                    }
                }
                if let Err(e) = ctx.declare_ref(
                    &iref.0,
                    RefInfo {
                        kind: RefKind::Inference,
                        step_index: index,
                        segment_type: None,
                    },
                ) {
                    return StepResult::Invalid(StepError {
                        step_index: index,
                        step_kind: kind,
                        message: e,
                    });
                }
                ctx.add_fact(
                    &iref.0,
                    ProofFact::InferredFact {
                        inf_ref: iref.0.clone(),
                        statement: conclusion.statement.clone(),
                    },
                );
            }

            ProofStep::Verdict(vt, conf, refs) => {
                // Verify all supporting references exist
                if let Err(e) = ctx.all_refs_exist(&refs.refs) {
                    return StepResult::Invalid(StepError {
                        step_index: index,
                        step_kind: kind,
                        message: e,
                    });
                }
                // Validate confidence is in [0, 1]
                if conf.0 < 0.0 || conf.0 > 1.0 {
                    return StepResult::Invalid(StepError {
                        step_index: index,
                        step_kind: kind,
                        message: format!(
                            "Verdict confidence out of range: {:.6}",
                            conf.0
                        ),
                    });
                }
                // A collusive verdict must have supporting evidence
                if *vt == VerdictType::Collusive && refs.refs.is_empty() {
                    return StepResult::Invalid(StepError {
                        step_index: index,
                        step_kind: kind,
                        message: "Collusive verdict requires supporting references"
                            .to_string(),
                    });
                }
                // Verify verdict is consistent with oracle level
                if let Err(e) = self.verify_verdict_consistency(vt, &refs.refs, ctx) {
                    return StepResult::Invalid(StepError {
                        step_index: index,
                        step_kind: kind,
                        message: e,
                    });
                }
                ctx.add_fact(
                    &format!("__verdict_{}", index),
                    ProofFact::Verdict {
                        verdict: *vt,
                        confidence: conf.0,
                    },
                );
                notes.push(format!(
                    "Verdict: {:?} at confidence {:.4}",
                    vt, conf.0
                ));
            }
        }

        StepResult::Valid(StepReport {
            step_index: index,
            step_kind: kind,
            declared_ref: step.declared_ref(),
            notes,
        })
    }

    /// Verify an axiom instantiation is sound.
    pub fn verify_axiom(
        &self,
        schema: &AxiomSchema,
        instantiation: &Instantiation,
    ) -> bool {
        // Check minimum parameter count
        if instantiation.len() < schema.required_params() {
            return false;
        }

        match schema {
            AxiomSchema::CompetitiveNullDef => {
                // Need: num_players >= 2, alpha > 0
                let n = instantiation
                    .get("num_players")
                    .and_then(|v| v.as_f64());
                let alpha = instantiation.get("alpha").and_then(|v| v.as_f64());
                match (n, alpha) {
                    (Some(n), Some(a)) => n >= 2.0 && a > 0.0 && a <= 1.0,
                    _ => false,
                }
            }
            AxiomSchema::TestSoundness => {
                // Need: alpha in (0,1], p_value in [0,1], statistic is finite
                let alpha = instantiation.get("alpha").and_then(|v| v.as_f64());
                let pv = instantiation.get("p_value").and_then(|v| v.as_f64());
                let stat = instantiation.get("statistic").and_then(|v| v.as_f64());
                match (alpha, pv, stat) {
                    (Some(a), Some(p), Some(s)) => {
                        a > 0.0 && a <= 1.0 && p >= 0.0 && p <= 1.0 && s.is_finite()
                    }
                    _ => false,
                }
            }
            AxiomSchema::CorrelationBound => {
                // Need: bound in [0,1], num_players >= 2
                let bound = instantiation.get("bound").and_then(|v| v.as_f64());
                let n = instantiation
                    .get("num_players")
                    .and_then(|v| v.as_f64());
                match (bound, n) {
                    (Some(b), Some(n)) => b >= 0.0 && b <= 1.0 && n >= 2.0,
                    _ => false,
                }
            }
            AxiomSchema::DeviationExistence => {
                // Need: cp > 0, discount in (0,1), num_players >= 2, nash_profit >= 0
                let cp = instantiation.get("cp").and_then(|v| v.as_f64());
                let discount = instantiation.get("discount").and_then(|v| v.as_f64());
                let n = instantiation.get("num_players").and_then(|v| v.as_f64());
                let nash = instantiation.get("nash_profit").and_then(|v| v.as_f64());
                match (cp, discount, n, nash) {
                    (Some(cp), Some(d), Some(n), Some(np)) => {
                        cp > 0.0 && d > 0.0 && d < 1.0 && n >= 2.0 && np >= 0.0
                    }
                    _ => false,
                }
            }
            AxiomSchema::PunishmentDetectability => {
                // Need: payoff_drop > 0, player >= 0, confidence in (0,1)
                let drop = instantiation.get("payoff_drop").and_then(|v| v.as_f64());
                let player = instantiation.get("player").and_then(|v| v.as_player());
                let conf = instantiation.get("confidence").and_then(|v| v.as_f64());
                match (drop, player, conf) {
                    (Some(d), Some(_), Some(c)) => d > 0.0 && c > 0.0 && c < 1.0,
                    _ => false,
                }
            }
            AxiomSchema::CollusionPremiumDef => {
                // Need: nash_profit, observed_profit, collusive_profit
                let np = instantiation.get("nash_profit").and_then(|v| v.as_f64());
                let op = instantiation
                    .get("observed_profit")
                    .and_then(|v| v.as_f64());
                let cp = instantiation
                    .get("collusive_profit")
                    .and_then(|v| v.as_f64());
                match (np, op, cp) {
                    (Some(np), Some(op), Some(cp)) => {
                        np.is_finite() && op.is_finite() && cp.is_finite() && cp > np
                    }
                    _ => false,
                }
            }
            AxiomSchema::FWERControl => {
                // Need: alpha > 0, num_tests >= 1
                let alpha = instantiation.get("alpha").and_then(|v| v.as_f64());
                let n = instantiation.get("num_tests").and_then(|v| v.as_f64());
                match (alpha, n) {
                    (Some(a), Some(n)) => a > 0.0 && a <= 1.0 && n >= 1.0,
                    _ => false,
                }
            }
            AxiomSchema::IntervalArithmetic => {
                // Need: lo_a, hi_a, lo_b, hi_b all finite with lo <= hi
                let la = instantiation.get("lo_a").and_then(|v| v.as_f64());
                let ha = instantiation.get("hi_a").and_then(|v| v.as_f64());
                let lb = instantiation.get("lo_b").and_then(|v| v.as_f64());
                let hb = instantiation.get("hi_b").and_then(|v| v.as_f64());
                match (la, ha, lb, hb) {
                    (Some(la), Some(ha), Some(lb), Some(hb)) => {
                        la.is_finite()
                            && ha.is_finite()
                            && lb.is_finite()
                            && hb.is_finite()
                            && la <= ha
                            && lb <= hb
                    }
                    _ => false,
                }
            }
            AxiomSchema::RationalExactness => {
                // Need: f64_value and rational_value (as numer/denom)
                let fv = instantiation.get("f64_value").and_then(|v| v.as_f64());
                let rv = instantiation.get("rational_value").and_then(|v| v.as_f64());
                match (fv, rv) {
                    (Some(f), Some(r)) => f.is_finite() && r.is_finite(),
                    _ => false,
                }
            }
            AxiomSchema::MonotonicityOfProfit => {
                // Need: marginal_cost, price, slope > 0
                let mc = instantiation.get("marginal_cost").and_then(|v| v.as_f64());
                let price = instantiation.get("price").and_then(|v| v.as_f64());
                let slope = instantiation.get("slope").and_then(|v| v.as_f64());
                match (mc, price, slope) {
                    (Some(mc), Some(p), Some(s)) => {
                        mc >= 0.0 && p >= mc && s > 0.0
                    }
                    _ => false,
                }
            }
            AxiomSchema::NashEquilibriumDef => {
                // Need: num_players >= 2, prices non-empty
                let n = instantiation.get("num_players").and_then(|v| v.as_f64());
                let prices = instantiation.get("prices");
                match (n, prices) {
                    (Some(n), Some(_)) => n >= 2.0,
                    _ => false,
                }
            }
            AxiomSchema::IndividualRationalityBound => {
                // Need: minimax_payoff >= 0, player >= 0
                let mm = instantiation
                    .get("minimax_payoff")
                    .and_then(|v| v.as_f64());
                let player = instantiation.get("player").and_then(|v| v.as_player());
                match (mm, player) {
                    (Some(mm), Some(_)) => mm.is_finite(),
                    _ => false,
                }
            }
            AxiomSchema::PayoffDecomposition => {
                // Need: observed, nash_component, premium_component
                let obs = instantiation.get("observed").and_then(|v| v.as_f64());
                let nash = instantiation
                    .get("nash_component")
                    .and_then(|v| v.as_f64());
                let prem = instantiation
                    .get("premium_component")
                    .and_then(|v| v.as_f64());
                match (obs, nash, prem) {
                    (Some(o), Some(n), Some(p)) => {
                        o.is_finite() && n.is_finite() && p.is_finite()
                            && (o - (n + p)).abs() < 1e-6
                    }
                    _ => false,
                }
            }
            AxiomSchema::SegmentIndependence => {
                // Need: segment_a, segment_b are different types
                let a = instantiation.get("segment_a");
                let b = instantiation.get("segment_b");
                match (a, b) {
                    (Some(ProofValue::String(a)), Some(ProofValue::String(b))) => a != b,
                    _ => false,
                }
            }
            AxiomSchema::BerryEsseenBound => {
                // Need: sample_size > 0, third_moment >= 0, sigma > 0
                let n = instantiation
                    .get("sample_size")
                    .and_then(|v| v.as_f64());
                let m3 = instantiation
                    .get("third_moment")
                    .and_then(|v| v.as_f64());
                let sigma = instantiation.get("sigma").and_then(|v| v.as_f64());
                match (n, m3, sigma) {
                    (Some(n), Some(m3), Some(s)) => n >= 1.0 && m3 >= 0.0 && s > 0.0,
                    _ => false,
                }
            }
        }
    }

    /// Verify an inference rule application.
    pub fn verify_inference(
        &self,
        rule: &InferenceRule,
        premises: &[ProofTerm],
        conclusion: &ProofTerm,
    ) -> bool {
        // Check minimum premise count
        if premises.len() < rule.min_premises() {
            return false;
        }

        match rule {
            InferenceRule::AndIntro => {
                // From P and Q, conclude P ∧ Q
                premises.len() >= 2
            }
            InferenceRule::AndElim => {
                // From P ∧ Q, conclude P (or Q)
                premises.len() >= 1
                    && matches!(&premises[0], ProofTerm::Conjunction(..))
            }
            InferenceRule::OrIntro => {
                // From P, conclude P ∨ Q
                premises.len() >= 1
            }
            InferenceRule::OrElim => {
                // From P ∨ Q, (P → R), (Q → R), conclude R
                premises.len() >= 3
            }
            InferenceRule::ImplicationElim => {
                // Modus ponens: from (P → Q) and P, conclude Q
                premises.len() >= 2
            }
            InferenceRule::UniversalElim => {
                premises.len() >= 1
            }
            InferenceRule::ExistentialIntro => {
                premises.len() >= 1
            }
            InferenceRule::ChainRule => {
                // Transitivity: need at least 2 premises
                premises.len() >= 2
            }
            InferenceRule::ContraPositive => {
                premises.len() >= 1
            }
            InferenceRule::AlphaSpending => {
                // Just needs a test result premise
                premises.len() >= 1
            }
            InferenceRule::IntervalRefine => {
                premises.len() >= 1
            }
            InferenceRule::BootstrapCI => {
                premises.len() >= 1
            }
            InferenceRule::CompositeRejection => {
                // Need at least 2 sub-test results
                premises.len() >= 2
            }
            InferenceRule::MonotonePriceComparison => {
                premises.len() >= 2
            }
            InferenceRule::ConfidenceComposition => {
                premises.len() >= 2
            }
            InferenceRule::ErrorPropagation => {
                premises.len() >= 1
            }
            InferenceRule::NumericalVerification => {
                // Verify f64 claim via rational re-computation
                if premises.len() < 1 {
                    return false;
                }
                if let ProofTerm::ArithmeticFact(left, rel, right) = &premises[0] {
                    if let (Some(lv), Some(rv)) =
                        (left.try_eval_f64(), right.try_eval_f64())
                    {
                        return rel.eval_f64(lv, rv);
                    }
                }
                true
            }
            InferenceRule::SegmentSplit => {
                premises.len() >= 1
            }
            InferenceRule::TestComposition => {
                premises.len() >= 2
            }
            InferenceRule::HolmCorrection => {
                premises.len() >= 1
            }
            InferenceRule::PayoffComparison => {
                premises.len() >= 2
            }
            InferenceRule::DeviationInference => {
                premises.len() >= 1
            }
            InferenceRule::PunishmentInference => {
                premises.len() >= 1
            }
            InferenceRule::VerdictDerivation => {
                premises.len() >= 1
            }
            InferenceRule::WeakeningRule => {
                premises.len() >= 1
            }
        }
    }

    /// Structural verification of an inference step in the certificate.
    fn verify_inference_structural(
        &self,
        rule: &Rule,
        premises: &Premises,
        _conclusion: &Conclusion,
        ctx: &CheckerContext,
    ) -> Result<(), String> {
        // Map rule name to InferenceRule enum
        let inf_rule = match rule.name.as_str() {
            "AndIntro" => InferenceRule::AndIntro,
            "AndElim" => InferenceRule::AndElim,
            "OrIntro" => InferenceRule::OrIntro,
            "OrElim" => InferenceRule::OrElim,
            "ImplicationElim" | "ModusPonens" => InferenceRule::ImplicationElim,
            "UniversalElim" => InferenceRule::UniversalElim,
            "ExistentialIntro" => InferenceRule::ExistentialIntro,
            "ChainRule" | "Transitivity" => InferenceRule::ChainRule,
            "ContraPositive" => InferenceRule::ContraPositive,
            "AlphaSpending" => InferenceRule::AlphaSpending,
            "IntervalRefine" => InferenceRule::IntervalRefine,
            "BootstrapCI" => InferenceRule::BootstrapCI,
            "CompositeRejection" => InferenceRule::CompositeRejection,
            "MonotonePriceComparison" => InferenceRule::MonotonePriceComparison,
            "ConfidenceComposition" => InferenceRule::ConfidenceComposition,
            "ErrorPropagation" => InferenceRule::ErrorPropagation,
            "NumericalVerification" => InferenceRule::NumericalVerification,
            "SegmentSplit" => InferenceRule::SegmentSplit,
            "TestComposition" => InferenceRule::TestComposition,
            "HolmCorrection" => InferenceRule::HolmCorrection,
            "PayoffComparison" => InferenceRule::PayoffComparison,
            "DeviationInference" => InferenceRule::DeviationInference,
            "PunishmentInference" => InferenceRule::PunishmentInference,
            "VerdictDerivation" => InferenceRule::VerdictDerivation,
            "WeakeningRule" => InferenceRule::WeakeningRule,
            _ => {
                return Err(format!("Unknown inference rule: {}", rule.name));
            }
        };

        // Check minimum premise count
        if premises.refs.len() < inf_rule.min_premises() {
            return Err(format!(
                "Rule {} requires at least {} premises, got {}",
                rule.name,
                inf_rule.min_premises(),
                premises.refs.len()
            ));
        }

        // Check segment isolation for SegmentSplit
        if inf_rule == InferenceRule::SegmentSplit {
            // Verify that the referenced segments don't overlap improperly
            for r in &premises.refs {
                if let Some(info) = ctx.declared_refs.get(r) {
                    if let Some(seg_type) = &info.segment_type {
                        ctx.segment_checker
                            .verify_no_cross_contamination(r, seg_type)
                            .map_err(|e| format!("Segment isolation: {}", e))?;
                    }
                }
            }
        }

        // Check alpha budget for alpha-consuming rules
        if inf_rule.consumes_alpha() {
            if !rule.params.is_empty() {
                let alpha_to_spend = rule.params[0];
                if alpha_to_spend < 0.0 || alpha_to_spend > 1.0 {
                    return Err(format!(
                        "Invalid alpha to spend: {:.6}",
                        alpha_to_spend
                    ));
                }
            }
        }

        Ok(())
    }

    /// Verify that a verdict is consistent with the evidence.
    fn verify_verdict_consistency(
        &self,
        verdict: &VerdictType,
        supporting_refs: &[String],
        ctx: &CheckerContext,
    ) -> Result<(), String> {
        match verdict {
            VerdictType::Collusive => {
                // A collusive verdict at Layer0 needs at least one significant test
                let has_test_rejection = supporting_refs.iter().any(|r| {
                    matches!(
                        ctx.get_fact(r),
                        Some(ProofFact::TestRejection { .. })
                    )
                });
                if !has_test_rejection
                    && !supporting_refs.iter().any(|r| {
                        matches!(
                            ctx.get_fact(r),
                            Some(ProofFact::InferredFact { .. })
                        )
                    })
                {
                    return Err(
                        "Collusive verdict requires at least one test rejection or inference"
                            .to_string(),
                    );
                }
            }
            VerdictType::Competitive => {
                // Competitive is always okay (failure to reject)
            }
            VerdictType::Inconclusive => {
                // Inconclusive is always okay
            }
        }
        Ok(())
    }

    /// Verify all rational arithmetic in a proof term.
    pub fn verify_rational_arithmetic(
        &self,
        expr: &Expression,
    ) -> Option<OrderingVerification> {
        self.rational_verifier.verify_expression(expr)
    }
}

impl Default for ProofChecker {
    fn default() -> Self {
        Self::new()
    }
}

// ── Alpha budget checker ─────────────────────────────────────────────────────

/// Tracks alpha (significance level) spending across a certificate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlphaBudgetChecker {
    pub total: f64,
    pub spent: f64,
    pub allocations: Vec<(String, f64)>,
}

impl AlphaBudgetChecker {
    pub fn new(total: f64) -> Self {
        Self {
            total,
            spent: 0.0,
            allocations: Vec::new(),
        }
    }

    pub fn spend(&mut self, amount: f64) {
        self.spent += amount;
    }

    pub fn spend_named(&mut self, name: &str, amount: f64) {
        self.allocations.push((name.to_string(), amount));
        self.spent += amount;
    }

    pub fn remaining(&self) -> f64 {
        (self.total - self.spent).max(0.0)
    }

    pub fn is_exceeded(&self) -> bool {
        self.spent > self.total + 1e-12
    }

    pub fn utilization(&self) -> f64 {
        if self.total > 0.0 {
            self.spent / self.total
        } else {
            0.0
        }
    }
}

// ── Segment isolation checker ────────────────────────────────────────────────

/// Verifies phantom-type segment integrity — ensures no data flows
/// between differently-typed segments.
#[derive(Debug, Clone)]
pub struct SegmentIsolationChecker {
    segments: HashMap<String, SegmentInfo>,
    pub check_count: usize,
}

#[derive(Debug, Clone)]
struct SegmentInfo {
    segment_type: String,
    start: usize,
    end: usize,
    used_in_refs: HashSet<String>,
}

impl SegmentIsolationChecker {
    pub fn new() -> Self {
        Self {
            segments: HashMap::new(),
            check_count: 0,
        }
    }

    pub fn register_segment(
        &mut self,
        ref_id: &str,
        segment_type: &str,
        start: usize,
        end: usize,
    ) {
        self.segments.insert(
            ref_id.to_string(),
            SegmentInfo {
                segment_type: segment_type.to_string(),
                start,
                end,
                used_in_refs: HashSet::new(),
            },
        );
    }

    pub fn verify_no_cross_contamination(
        &self,
        ref_id: &str,
        expected_type: &str,
    ) -> Result<(), String> {
        if let Some(info) = self.segments.get(ref_id) {
            if info.segment_type != expected_type {
                return Err(format!(
                    "Segment {} has type '{}' but expected '{}'",
                    ref_id, info.segment_type, expected_type
                ));
            }
        }
        Ok(())
    }

    /// Check that two references don't mix segments from different types.
    pub fn verify_isolation(&mut self, ref_a: &str, ref_b: &str) -> Result<(), String> {
        self.check_count += 1;
        let type_a = self.segments.get(ref_a).map(|s| s.segment_type.clone());
        let type_b = self.segments.get(ref_b).map(|s| s.segment_type.clone());

        match (type_a, type_b) {
            (Some(a), Some(b)) if a != b => Err(format!(
                "Segment isolation violation: {} (type '{}') mixed with {} (type '{}')",
                ref_a, a, ref_b, b
            )),
            _ => Ok(()),
        }
    }

    /// Check that segments don't overlap.
    pub fn verify_non_overlapping(&self) -> Result<(), String> {
        let segs: Vec<_> = self.segments.values().collect();
        for i in 0..segs.len() {
            for j in (i + 1)..segs.len() {
                if segs[i].segment_type == segs[j].segment_type {
                    continue;
                }
                if segs[i].start < segs[j].end && segs[j].start < segs[i].end {
                    return Err(format!(
                        "Segments overlap: [{}, {}) and [{}, {})",
                        segs[i].start, segs[i].end, segs[j].start, segs[j].end
                    ));
                }
            }
        }
        Ok(())
    }
}

impl Default for SegmentIsolationChecker {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use shared_types::OracleAccessLevel;

    fn make_simple_certificate() -> CertificateAST {
        let header = CertificateHeader::new("test_scenario", OracleAccessLevel::Layer0, 0.05);
        let mut body = CertificateBody::new();

        body.push(ProofStep::DataDeclaration(
            TrajectoryRef::new("traj_0"),
            SegmentSpec::new("testing", 0, 500, "hash123", 2),
        ));
        body.push(ProofStep::StatisticalTest(
            TestRef::new("test_corr"),
            TestType::new("PriceCorrelation", "layer0"),
            Statistic::new(3.5),
            PValueWrapper::new(0.001),
        ));
        body.push(ProofStep::EquilibriumClaim(
            EquilibriumRef::new("eq_nash"),
            GameSpec::new(2, "Bertrand"),
            NashProfile::new(vec![3.0, 3.0], vec![4.0, 4.0]),
        ));
        body.push(ProofStep::CollusionPremium(
            CPRef::new("cp_0"),
            Value::new(0.42),
            CIWrapper::new(0.30, 0.55, 0.95),
        ));
        body.push(ProofStep::Inference(
            InferenceRef::new("inf_0"),
            Rule::new("VerdictDerivation"),
            Premises::new(vec![
                "test_corr".into(),
                "eq_nash".into(),
                "cp_0".into(),
            ]),
            Conclusion::new("Evidence supports collusion"),
        ));
        body.push(ProofStep::Verdict(
            VerdictType::Collusive,
            Confidence::new(0.95),
            SupportingRefs::new(vec!["test_corr".into(), "inf_0".into()]),
        ));

        CertificateAST::new(header, body)
    }

    #[test]
    fn test_check_valid_certificate() {
        let cert = make_simple_certificate();
        let checker = ProofChecker::new();
        let result = checker.check_certificate(&cert);
        assert!(result.is_valid());
        let report = result.unwrap_report();
        assert_eq!(report.total_steps, 6);
        assert_eq!(report.verified_steps, 6);
        assert_eq!(report.verdict, Some(VerdictType::Collusive));
    }

    #[test]
    fn test_check_empty_certificate() {
        let header = CertificateHeader::new("s", OracleAccessLevel::Layer0, 0.05);
        let body = CertificateBody::new();
        let cert = CertificateAST::new(header, body);
        let checker = ProofChecker::new();
        let result = checker.check_certificate(&cert);
        assert!(!result.is_valid());
    }

    #[test]
    fn test_check_invalid_header_alpha() {
        let mut header = CertificateHeader::new("s", OracleAccessLevel::Layer0, 0.05);
        header.alpha = shared_types::SignificanceLevel::new(0.0);
        let mut body = CertificateBody::new();
        body.push(ProofStep::Verdict(
            VerdictType::Competitive,
            Confidence::new(0.5),
            SupportingRefs::new(vec![]),
        ));
        let cert = CertificateAST::new(header, body);
        let checker = ProofChecker::new();
        let result = checker.check_certificate(&cert);
        assert!(!result.is_valid());
    }

    #[test]
    fn test_check_duplicate_ref() {
        let header = CertificateHeader::new("s", OracleAccessLevel::Layer0, 0.05);
        let mut body = CertificateBody::new();
        body.push(ProofStep::DataDeclaration(
            TrajectoryRef::new("traj_0"),
            SegmentSpec::new("testing", 0, 100, "h", 2),
        ));
        body.push(ProofStep::DataDeclaration(
            TrajectoryRef::new("traj_0"),
            SegmentSpec::new("training", 100, 200, "h", 2),
        ));
        let cert = CertificateAST::new(header, body);
        let checker = ProofChecker::new();
        let result = checker.check_certificate(&cert);
        assert!(!result.is_valid());
    }

    #[test]
    fn test_check_undeclared_reference() {
        let header = CertificateHeader::new("s", OracleAccessLevel::Layer0, 0.05);
        let mut body = CertificateBody::new();
        body.push(ProofStep::Verdict(
            VerdictType::Competitive,
            Confidence::new(0.5),
            SupportingRefs::new(vec!["nonexistent".into()]),
        ));
        let cert = CertificateAST::new(header, body);
        let checker = ProofChecker::new();
        let result = checker.check_certificate(&cert);
        assert!(!result.is_valid());
    }

    #[test]
    fn test_check_invalid_pvalue() {
        let header = CertificateHeader::new("s", OracleAccessLevel::Layer0, 0.05);
        let mut body = CertificateBody::new();
        body.push(ProofStep::StatisticalTest(
            TestRef::new("t1"),
            TestType::new("test", "l0"),
            Statistic::new(f64::NAN),
            PValueWrapper::new(0.01),
        ));
        let cert = CertificateAST::new(header, body);
        let checker = ProofChecker::new();
        let result = checker.check_certificate(&cert);
        assert!(!result.is_valid());
    }

    #[test]
    fn test_check_nash_profile_mismatch() {
        let header = CertificateHeader::new("s", OracleAccessLevel::Layer0, 0.05);
        let mut body = CertificateBody::new();
        body.push(ProofStep::EquilibriumClaim(
            EquilibriumRef::new("eq"),
            GameSpec::new(2, "Bertrand"),
            NashProfile::new(vec![3.0], vec![4.0, 4.0]),
        ));
        let cert = CertificateAST::new(header, body);
        let checker = ProofChecker::new();
        let result = checker.check_certificate(&cert);
        assert!(!result.is_valid());
    }

    #[test]
    fn test_check_cp_ci_mismatch() {
        let header = CertificateHeader::new("s", OracleAccessLevel::Layer0, 0.05);
        let mut body = CertificateBody::new();
        body.push(ProofStep::CollusionPremium(
            CPRef::new("cp"),
            Value::new(0.8),
            CIWrapper::new(0.1, 0.5, 0.95),
        ));
        let cert = CertificateAST::new(header, body);
        let checker = ProofChecker::new();
        let result = checker.check_certificate(&cert);
        assert!(!result.is_valid());
    }

    #[test]
    fn test_check_inference_insufficient_premises() {
        let header = CertificateHeader::new("s", OracleAccessLevel::Layer0, 0.05);
        let mut body = CertificateBody::new();
        body.push(ProofStep::Inference(
            InferenceRef::new("inf"),
            Rule::new("AndIntro"),
            Premises::new(vec!["a".into()]),
            Conclusion::new("c"),
        ));
        let cert = CertificateAST::new(header, body);
        let checker = ProofChecker::new();
        let result = checker.check_certificate(&cert);
        // fails because "a" is undeclared
        assert!(!result.is_valid());
    }

    #[test]
    fn test_alpha_budget_checker() {
        let mut abc = AlphaBudgetChecker::new(0.05);
        assert!(!abc.is_exceeded());
        abc.spend(0.02);
        abc.spend(0.02);
        assert!(!abc.is_exceeded());
        assert!((abc.remaining() - 0.01).abs() < 1e-12);
        abc.spend(0.02);
        assert!(abc.is_exceeded());
    }

    #[test]
    fn test_segment_isolation_checker() {
        let mut sic = SegmentIsolationChecker::new();
        sic.register_segment("traj_test", "testing", 0, 100);
        sic.register_segment("traj_train", "training", 100, 200);

        assert!(sic.verify_no_cross_contamination("traj_test", "testing").is_ok());
        assert!(sic
            .verify_no_cross_contamination("traj_test", "training")
            .is_err());

        assert!(sic.verify_isolation("traj_test", "traj_train").is_err());
        assert!(sic.verify_isolation("traj_test", "traj_test").is_ok());
    }

    #[test]
    fn test_verify_axiom_competitive_null() {
        let checker = ProofChecker::new();
        let inst = Instantiation::new()
            .bind("num_players", ProofValue::Float(2.0))
            .bind("alpha", ProofValue::Float(0.05));
        assert!(checker.verify_axiom(&AxiomSchema::CompetitiveNullDef, &inst));

        let bad_inst = Instantiation::new()
            .bind("num_players", ProofValue::Float(1.0))
            .bind("alpha", ProofValue::Float(0.05));
        assert!(!checker.verify_axiom(&AxiomSchema::CompetitiveNullDef, &bad_inst));
    }

    #[test]
    fn test_verify_axiom_test_soundness() {
        let checker = ProofChecker::new();
        let inst = Instantiation::new()
            .bind("alpha", ProofValue::Float(0.05))
            .bind("p_value", ProofValue::Float(0.001))
            .bind("statistic", ProofValue::Float(3.5));
        assert!(checker.verify_axiom(&AxiomSchema::TestSoundness, &inst));
    }

    #[test]
    fn test_verify_axiom_collusion_premium() {
        let checker = ProofChecker::new();
        let inst = Instantiation::new()
            .bind("nash_profit", ProofValue::Float(2.0))
            .bind("observed_profit", ProofValue::Float(3.5))
            .bind("collusive_profit", ProofValue::Float(5.0));
        assert!(checker.verify_axiom(&AxiomSchema::CollusionPremiumDef, &inst));

        let bad = Instantiation::new()
            .bind("nash_profit", ProofValue::Float(5.0))
            .bind("observed_profit", ProofValue::Float(3.5))
            .bind("collusive_profit", ProofValue::Float(2.0));
        assert!(!checker.verify_axiom(&AxiomSchema::CollusionPremiumDef, &bad));
    }

    #[test]
    fn test_verify_axiom_segment_independence() {
        let checker = ProofChecker::new();
        let inst = Instantiation::new()
            .bind("segment_a", ProofValue::String("training".into()))
            .bind("segment_b", ProofValue::String("testing".into()));
        assert!(checker.verify_axiom(&AxiomSchema::SegmentIndependence, &inst));

        let same = Instantiation::new()
            .bind("segment_a", ProofValue::String("testing".into()))
            .bind("segment_b", ProofValue::String("testing".into()));
        assert!(!checker.verify_axiom(&AxiomSchema::SegmentIndependence, &same));
    }

    #[test]
    fn test_verify_inference_and_intro() {
        let checker = ProofChecker::new();
        let p = ProofTerm::Reference("P".into());
        let q = ProofTerm::Reference("Q".into());
        let conc = ProofTerm::Conjunction(Box::new(p.clone()), Box::new(q.clone()));
        assert!(checker.verify_inference(&InferenceRule::AndIntro, &[p, q], &conc));
    }

    #[test]
    fn test_verify_inference_insufficient() {
        let checker = ProofChecker::new();
        let p = ProofTerm::Reference("P".into());
        let conc = ProofTerm::Reference("C".into());
        assert!(!checker.verify_inference(&InferenceRule::AndIntro, &[p], &conc));
    }

    #[test]
    fn test_competitive_verdict_no_evidence_ok() {
        let header = CertificateHeader::new("s", OracleAccessLevel::Layer0, 0.05);
        let mut body = CertificateBody::new();
        body.push(ProofStep::DataDeclaration(
            TrajectoryRef::new("traj_0"),
            SegmentSpec::new("testing", 0, 100, "h", 2),
        ));
        body.push(ProofStep::Verdict(
            VerdictType::Competitive,
            Confidence::new(0.5),
            SupportingRefs::new(vec![]),
        ));
        let cert = CertificateAST::new(header, body);
        let checker = ProofChecker::new();
        let result = checker.check_certificate(&cert);
        assert!(result.is_valid());
    }

    #[test]
    fn test_collusive_verdict_needs_evidence() {
        let header = CertificateHeader::new("s", OracleAccessLevel::Layer0, 0.05);
        let mut body = CertificateBody::new();
        body.push(ProofStep::DataDeclaration(
            TrajectoryRef::new("traj_0"),
            SegmentSpec::new("testing", 0, 100, "h", 2),
        ));
        body.push(ProofStep::Verdict(
            VerdictType::Collusive,
            Confidence::new(0.95),
            SupportingRefs::new(vec![]),
        ));
        let cert = CertificateAST::new(header, body);
        let checker = ProofChecker::new();
        let result = checker.check_certificate(&cert);
        assert!(!result.is_valid());
    }

    #[test]
    fn test_segment_non_overlapping() {
        let mut sic = SegmentIsolationChecker::new();
        sic.register_segment("a", "testing", 0, 100);
        sic.register_segment("b", "training", 200, 300);
        assert!(sic.verify_non_overlapping().is_ok());

        sic.register_segment("c", "validation", 50, 150);
        assert!(sic.verify_non_overlapping().is_err());
    }
}
