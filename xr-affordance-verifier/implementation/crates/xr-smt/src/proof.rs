//! Proof objects for SMT-based accessibility verification.
//!
//! This module provides structured proof representations that record the
//! reasoning chain produced by the SMT solver. Proofs can be validated,
//! serialized, and bundled into [`ProofCertificate`]s that pair a proof
//! with the spatial region it covers.

use serde::{Deserialize, Serialize};
use uuid::Uuid;
use xr_types::VerifierError;

use crate::expr::SmtExpr;

// ---------------------------------------------------------------------------
// ProofStep
// ---------------------------------------------------------------------------

/// A single step in an SMT proof derivation.
///
/// Each variant carries a unique `id` (its index in the proof's step list)
/// and may reference earlier steps by their IDs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProofStep {
    /// A given assertion (no dependencies).
    Axiom {
        /// Unique step identifier.
        id: usize,
        /// The asserted formula.
        formula: SmtExpr,
    },

    /// Resolution of two clauses on a pivot variable.
    Resolution {
        /// Unique step identifier.
        id: usize,
        /// Step ID of the left premise.
        left: usize,
        /// Step ID of the right premise.
        right: usize,
        /// The variable resolved upon.
        pivot: String,
        /// The resulting clause.
        result: SmtExpr,
    },

    /// A lemma derived from a background theory (e.g. QF_LRA).
    TheoryLemma {
        /// Unique step identifier.
        id: usize,
        /// Name of the background theory.
        theory: String,
        /// The derived formula.
        formula: SmtExpr,
        /// Human-readable justification.
        justification: String,
    },

    /// A formula assumed for the sake of deriving a contradiction.
    Hypothesis {
        /// Unique step identifier.
        id: usize,
        /// The hypothesised formula.
        formula: SmtExpr,
    },

    /// Modus ponens: from A and A → B derive B.
    ModusPonens {
        /// Unique step identifier.
        id: usize,
        /// Step ID providing the antecedent A.
        antecedent: usize,
        /// Step ID providing the implication A → B.
        implication: usize,
        /// The derived consequent B.
        result: SmtExpr,
    },

    /// Explicit contradiction between two complementary steps (P and ¬P).
    Contradiction {
        /// Unique step identifier.
        id: usize,
        /// Step ID of P.
        left: usize,
        /// Step ID of ¬P.
        right: usize,
    },
}

impl ProofStep {
    /// Return the unique identifier of this step.
    pub fn id(&self) -> usize {
        match self {
            ProofStep::Axiom { id, .. }
            | ProofStep::Resolution { id, .. }
            | ProofStep::TheoryLemma { id, .. }
            | ProofStep::Hypothesis { id, .. }
            | ProofStep::ModusPonens { id, .. }
            | ProofStep::Contradiction { id, .. } => *id,
        }
    }

    /// Return the formula associated with this step, if any.
    ///
    /// [`Contradiction`](ProofStep::Contradiction) steps have no formula.
    pub fn formula(&self) -> Option<&SmtExpr> {
        match self {
            ProofStep::Axiom { formula, .. }
            | ProofStep::Hypothesis { formula, .. }
            | ProofStep::TheoryLemma { formula, .. } => Some(formula),
            ProofStep::Resolution { result, .. }
            | ProofStep::ModusPonens { result, .. } => Some(result),
            ProofStep::Contradiction { .. } => None,
        }
    }

    /// Return the IDs of proof steps this step depends on.
    ///
    /// Leaf steps ([`Axiom`](ProofStep::Axiom), [`Hypothesis`](ProofStep::Hypothesis),
    /// and [`TheoryLemma`](ProofStep::TheoryLemma)) have no dependencies.
    pub fn dependencies(&self) -> Vec<usize> {
        match self {
            ProofStep::Axiom { .. }
            | ProofStep::Hypothesis { .. }
            | ProofStep::TheoryLemma { .. } => vec![],
            ProofStep::Resolution { left, right, .. }
            | ProofStep::Contradiction { left, right, .. } => vec![*left, *right],
            ProofStep::ModusPonens {
                antecedent,
                implication,
                ..
            } => vec![*antecedent, *implication],
        }
    }

    /// Whether this step is a leaf (no dependencies on other proof steps).
    pub fn is_leaf(&self) -> bool {
        matches!(
            self,
            ProofStep::Axiom { .. }
                | ProofStep::Hypothesis { .. }
                | ProofStep::TheoryLemma { .. }
        )
    }
}

// ---------------------------------------------------------------------------
// ProofConclusion / ProofMetadata
// ---------------------------------------------------------------------------

/// Overall outcome of the SMT check.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProofConclusion {
    /// The assertions are unsatisfiable (property verified).
    Unsat,
    /// The assertions are satisfiable (counter-example exists).
    Sat,
    /// The solver could not determine satisfiability.
    Unknown,
}

/// Metadata attached to a proof.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofMetadata {
    /// Unique identifier for this proof.
    pub proof_id: Uuid,
    /// Name of the solver that produced the proof.
    pub solver_name: String,
    /// Wall-clock time taken to produce the proof (seconds).
    pub creation_time_secs: f64,
    /// Number of distinct variables in the encoding.
    pub num_variables: usize,
    /// Number of top-level assertions fed to the solver.
    pub num_assertions: usize,
}

impl Default for ProofMetadata {
    fn default() -> Self {
        Self {
            proof_id: Uuid::new_v4(),
            solver_name: String::from("unknown"),
            creation_time_secs: 0.0,
            num_variables: 0,
            num_assertions: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// SmtProof
// ---------------------------------------------------------------------------

/// A complete SMT proof: an ordered sequence of [`ProofStep`]s together with
/// a conclusion and optional metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmtProof {
    /// Ordered proof steps (index == step id).
    steps: Vec<ProofStep>,
    /// The overall conclusion.
    pub conclusion: ProofConclusion,
    /// Proof metadata.
    pub metadata: ProofMetadata,
}

impl SmtProof {
    /// Create an empty proof with the given conclusion.
    pub fn new(conclusion: ProofConclusion) -> Self {
        Self {
            steps: Vec::new(),
            conclusion,
            metadata: ProofMetadata::default(),
        }
    }

    /// Attach metadata (builder-style).
    pub fn with_metadata(mut self, meta: ProofMetadata) -> Self {
        self.metadata = meta;
        self
    }

    // -- step insertion helpers ---------------------------------------------

    /// Add an axiom step and return its id.
    pub fn add_axiom(&mut self, formula: SmtExpr) -> usize {
        let id = self.steps.len();
        self.steps.push(ProofStep::Axiom { id, formula });
        id
    }

    /// Add a resolution step and return its id.
    pub fn add_resolution(
        &mut self,
        left: usize,
        right: usize,
        pivot: String,
        result: SmtExpr,
    ) -> usize {
        let id = self.steps.len();
        self.steps.push(ProofStep::Resolution {
            id,
            left,
            right,
            pivot,
            result,
        });
        id
    }

    /// Add a theory-lemma step and return its id.
    pub fn add_theory_lemma(
        &mut self,
        theory: &str,
        formula: SmtExpr,
        justification: &str,
    ) -> usize {
        let id = self.steps.len();
        self.steps.push(ProofStep::TheoryLemma {
            id,
            theory: theory.to_owned(),
            formula,
            justification: justification.to_owned(),
        });
        id
    }

    /// Add a hypothesis step and return its id.
    pub fn add_hypothesis(&mut self, formula: SmtExpr) -> usize {
        let id = self.steps.len();
        self.steps.push(ProofStep::Hypothesis { id, formula });
        id
    }

    /// Add a modus-ponens step and return its id.
    pub fn add_modus_ponens(
        &mut self,
        antecedent: usize,
        implication: usize,
        result: SmtExpr,
    ) -> usize {
        let id = self.steps.len();
        self.steps.push(ProofStep::ModusPonens {
            id,
            antecedent,
            implication,
            result,
        });
        id
    }

    /// Add a contradiction step and return its id.
    pub fn add_contradiction(&mut self, left: usize, right: usize) -> usize {
        let id = self.steps.len();
        self.steps.push(ProofStep::Contradiction { id, left, right });
        id
    }

    // -- accessors ----------------------------------------------------------

    /// Total number of proof steps.
    pub fn num_steps(&self) -> usize {
        self.steps.len()
    }

    /// Retrieve a step by its id, if it exists.
    pub fn get_step(&self, id: usize) -> Option<&ProofStep> {
        self.steps.get(id)
    }

    // -- validation ---------------------------------------------------------

    /// Validate structural integrity of the proof.
    ///
    /// Checks performed:
    /// - Every step id equals its position in the step vector.
    /// - All referenced step IDs are strictly less than the referencing step
    ///   (no forward references).
    /// - Leaf steps have no dependencies.
    /// - Non-leaf steps reference valid, existing steps.
    pub fn validate(&self) -> Result<(), VerifierError> {
        for (idx, step) in self.steps.iter().enumerate() {
            // ID must match position.
            if step.id() != idx {
                return Err(VerifierError::CertificateGeneration(format!(
                    "proof step at index {} has id {}",
                    idx,
                    step.id()
                )));
            }

            let deps = step.dependencies();

            // Leaf steps must have no dependencies.
            if step.is_leaf() && !deps.is_empty() {
                return Err(VerifierError::CertificateGeneration(format!(
                    "leaf step {} unexpectedly has dependencies",
                    idx
                )));
            }

            for &dep in &deps {
                // No forward references.
                if dep >= idx {
                    return Err(VerifierError::CertificateGeneration(format!(
                        "step {} references future step {}",
                        idx, dep
                    )));
                }
                // Referenced step must exist.
                if dep >= self.steps.len() {
                    return Err(VerifierError::CertificateGeneration(format!(
                        "step {} references non-existent step {}",
                        idx, dep
                    )));
                }
            }
        }

        Ok(())
    }

    // -- analysis -----------------------------------------------------------

    /// Maximum dependency-chain length (longest path from a leaf to the root).
    ///
    /// An empty proof has depth 0. A proof consisting only of leaf steps has
    /// depth 1.
    pub fn depth(&self) -> usize {
        if self.steps.is_empty() {
            return 0;
        }

        // depth[i] = 1 + max(depth[dep] for dep in deps), or 1 for leaves.
        let mut depths: Vec<usize> = Vec::with_capacity(self.steps.len());
        for step in &self.steps {
            let d = step
                .dependencies()
                .iter()
                .filter_map(|&dep| depths.get(dep).copied())
                .max()
                .map_or(1, |m| m + 1);
            depths.push(d);
        }

        depths.into_iter().max().unwrap_or(0)
    }

    /// Number of leaf steps (axioms, hypotheses, and theory lemmas).
    pub fn leaf_count(&self) -> usize {
        self.steps.iter().filter(|s| s.is_leaf()).count()
    }
}

// ---------------------------------------------------------------------------
// CertificateVerdict / ProofCertificate
// ---------------------------------------------------------------------------

/// Verdict of a certificate over a spatial region.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CertificateVerdict {
    /// The property was verified in the region.
    Verified,
    /// The property was refuted in the region.
    Refuted,
    /// The solver could not determine the verdict.
    Inconclusive,
}

/// A certificate that pairs an SMT proof with the spatial region it covers,
/// augmented with confidence and linearization error bounds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofCertificate {
    /// Unique certificate identifier.
    pub certificate_id: Uuid,
    /// The underlying SMT proof.
    pub proof: SmtProof,
    /// Lower bounds of the spatial region (one per dimension).
    pub region_lower: Vec<f64>,
    /// Upper bounds of the spatial region (one per dimension).
    pub region_upper: Vec<f64>,
    /// The certification verdict.
    pub verdict: CertificateVerdict,
    /// Confidence level in [0, 1].
    pub confidence: f64,
    /// Upper bound on linearization error across the region.
    pub linearization_error_bound: f64,
}

impl ProofCertificate {
    /// Create a new certificate with default confidence and error bound.
    pub fn new(
        proof: SmtProof,
        lower: Vec<f64>,
        upper: Vec<f64>,
        verdict: CertificateVerdict,
    ) -> Self {
        Self {
            certificate_id: Uuid::new_v4(),
            proof,
            region_lower: lower,
            region_upper: upper,
            verdict,
            confidence: 1.0,
            linearization_error_bound: 0.0,
        }
    }

    /// Set the confidence level (builder-style).
    pub fn with_confidence(mut self, conf: f64) -> Self {
        self.confidence = conf;
        self
    }

    /// Set the linearization error bound (builder-style).
    pub fn with_linearization_bound(mut self, bound: f64) -> Self {
        self.linearization_error_bound = bound;
        self
    }

    /// Whether the verdict is [`CertificateVerdict::Verified`].
    pub fn is_verified(&self) -> bool {
        self.verdict == CertificateVerdict::Verified
    }

    /// Compute the hyper-volume of the certified region.
    ///
    /// Returns the product of `(upper[i] - lower[i])` across all dimensions.
    /// If the bounds vectors have different lengths the shorter length is used.
    pub fn region_volume(&self) -> f64 {
        self.region_lower
            .iter()
            .zip(self.region_upper.iter())
            .map(|(lo, hi)| hi - lo)
            .product()
    }

    /// Serialize the certificate to a JSON string.
    pub fn to_json(&self) -> Result<String, VerifierError> {
        serde_json::to_string_pretty(self).map_err(|e| {
            VerifierError::CertificateGeneration(format!("JSON serialization failed: {e}"))
        })
    }
}

// ---------------------------------------------------------------------------
// ProofBuilder
// ---------------------------------------------------------------------------

/// Convenience builder for constructing and validating proofs step-by-step.
pub struct ProofBuilder {
    proof: SmtProof,
}

impl ProofBuilder {
    /// Start building a new proof with the given conclusion.
    pub fn new(conclusion: ProofConclusion) -> Self {
        Self {
            proof: SmtProof::new(conclusion),
        }
    }

    /// Attach metadata.
    pub fn with_metadata(mut self, meta: ProofMetadata) -> Self {
        self.proof.metadata = meta;
        self
    }

    /// Add an axiom step.
    pub fn add_axiom(&mut self, formula: SmtExpr) -> usize {
        self.proof.add_axiom(formula)
    }

    /// Add a resolution step.
    pub fn add_resolution(
        &mut self,
        left: usize,
        right: usize,
        pivot: String,
        result: SmtExpr,
    ) -> usize {
        self.proof.add_resolution(left, right, pivot, result)
    }

    /// Add a theory lemma step.
    pub fn add_theory_lemma(
        &mut self,
        theory: &str,
        formula: SmtExpr,
        justification: &str,
    ) -> usize {
        self.proof.add_theory_lemma(theory, formula, justification)
    }

    /// Add a hypothesis step.
    pub fn add_hypothesis(&mut self, formula: SmtExpr) -> usize {
        self.proof.add_hypothesis(formula)
    }

    /// Add a modus-ponens step.
    pub fn add_modus_ponens(
        &mut self,
        antecedent: usize,
        implication: usize,
        result: SmtExpr,
    ) -> usize {
        self.proof.add_modus_ponens(antecedent, implication, result)
    }

    /// Add a contradiction step.
    pub fn add_contradiction(&mut self, left: usize, right: usize) -> usize {
        self.proof.add_contradiction(left, right)
    }

    /// Validate the accumulated proof and return it if valid.
    pub fn build(self) -> Result<SmtProof, VerifierError> {
        self.proof.validate()?;
        Ok(self.proof)
    }
}

// ---------------------------------------------------------------------------
// ProofSummary / ProofSerializer
// ---------------------------------------------------------------------------

/// Compact summary of a proof's structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofSummary {
    /// Total number of steps.
    pub num_steps: usize,
    /// Number of axiom steps.
    pub num_axioms: usize,
    /// Number of theory-lemma steps.
    pub num_theory_lemmas: usize,
    /// Maximum dependency-chain length.
    pub depth: usize,
    /// Stringified conclusion.
    pub conclusion: String,
}

/// Serialization helpers for proofs.
pub struct ProofSerializer;

impl ProofSerializer {
    /// Serialize a proof to a JSON string.
    pub fn to_json(proof: &SmtProof) -> Result<String, VerifierError> {
        serde_json::to_string_pretty(proof).map_err(|e| {
            VerifierError::SmtEncoding(format!("proof JSON serialization failed: {e}"))
        })
    }

    /// Render a human-readable SMT-LIB2-style proof trace.
    pub fn to_smtlib2_proof(proof: &SmtProof) -> String {
        let mut buf = String::new();
        buf.push_str("; --- SMT proof trace ---\n");
        buf.push_str(&format!(
            "; conclusion: {}\n",
            conclusion_str(&proof.conclusion)
        ));
        buf.push_str(&format!("; steps: {}\n", proof.num_steps()));
        buf.push('\n');

        for step in &proof.steps {
            match step {
                ProofStep::Axiom { id, formula } => {
                    buf.push_str(&format!(
                        "(step @{} :rule axiom :clause {})\n",
                        id,
                        formula.to_smtlib2()
                    ));
                }
                ProofStep::Resolution {
                    id,
                    left,
                    right,
                    pivot,
                    result,
                } => {
                    buf.push_str(&format!(
                        "(step @{} :rule resolution :premises (@{} @{}) :pivot {} :clause {})\n",
                        id,
                        left,
                        right,
                        pivot,
                        result.to_smtlib2()
                    ));
                }
                ProofStep::TheoryLemma {
                    id,
                    theory,
                    formula,
                    justification,
                } => {
                    buf.push_str(&format!(
                        "(step @{} :rule theory_lemma :theory {} :justification \"{}\" :clause {})\n",
                        id,
                        theory,
                        justification,
                        formula.to_smtlib2()
                    ));
                }
                ProofStep::Hypothesis { id, formula } => {
                    buf.push_str(&format!(
                        "(step @{} :rule hypothesis :clause {})\n",
                        id,
                        formula.to_smtlib2()
                    ));
                }
                ProofStep::ModusPonens {
                    id,
                    antecedent,
                    implication,
                    result,
                } => {
                    buf.push_str(&format!(
                        "(step @{} :rule modus_ponens :premises (@{} @{}) :clause {})\n",
                        id,
                        antecedent,
                        implication,
                        result.to_smtlib2()
                    ));
                }
                ProofStep::Contradiction { id, left, right } => {
                    buf.push_str(&format!(
                        "(step @{} :rule contradiction :premises (@{} @{}))\n",
                        id, left, right
                    ));
                }
            }
        }

        buf
    }

    /// Produce a compact [`ProofSummary`].
    pub fn summary(proof: &SmtProof) -> ProofSummary {
        let num_axioms = proof
            .steps
            .iter()
            .filter(|s| matches!(s, ProofStep::Axiom { .. }))
            .count();
        let num_theory_lemmas = proof
            .steps
            .iter()
            .filter(|s| matches!(s, ProofStep::TheoryLemma { .. }))
            .count();

        ProofSummary {
            num_steps: proof.num_steps(),
            num_axioms,
            num_theory_lemmas,
            depth: proof.depth(),
            conclusion: conclusion_str(&proof.conclusion),
        }
    }
}

/// Stringify a [`ProofConclusion`].
fn conclusion_str(c: &ProofConclusion) -> String {
    match c {
        ProofConclusion::Unsat => "unsat".to_owned(),
        ProofConclusion::Sat => "sat".to_owned(),
        ProofConclusion::Unknown => "unknown".to_owned(),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: `x >= 0`.
    fn x_geq_zero() -> SmtExpr {
        SmtExpr::Ge(
            Box::new(SmtExpr::Var("x".into())),
            Box::new(SmtExpr::Const(0.0)),
        )
    }

    /// Helper: `x < 0`.
    fn x_lt_zero() -> SmtExpr {
        SmtExpr::Lt(
            Box::new(SmtExpr::Var("x".into())),
            Box::new(SmtExpr::Const(0.0)),
        )
    }

    /// Helper: `y > 1`.
    fn y_gt_one() -> SmtExpr {
        SmtExpr::Gt(
            Box::new(SmtExpr::Var("y".into())),
            Box::new(SmtExpr::Const(1.0)),
        )
    }

    /// Helper: `false`.
    fn smt_false() -> SmtExpr {
        SmtExpr::BoolConst(false)
    }

    // -----------------------------------------------------------------------

    #[test]
    fn test_proof_construction() {
        let mut proof = SmtProof::new(ProofConclusion::Unsat);

        let a0 = proof.add_axiom(x_geq_zero());
        let a1 = proof.add_axiom(x_lt_zero());
        let r = proof.add_resolution(a0, a1, "x".into(), smt_false());

        assert_eq!(proof.num_steps(), 3);
        assert_eq!(a0, 0);
        assert_eq!(a1, 1);
        assert_eq!(r, 2);

        let step = proof.get_step(r).unwrap();
        assert!(!step.is_leaf());
        assert_eq!(step.dependencies(), vec![0, 1]);
    }

    #[test]
    fn test_proof_validation_valid() {
        let mut proof = SmtProof::new(ProofConclusion::Unsat);
        proof.add_axiom(x_geq_zero());
        proof.add_axiom(x_lt_zero());
        proof.add_resolution(0, 1, "x".into(), smt_false());

        assert!(proof.validate().is_ok());
    }

    #[test]
    fn test_proof_validation_invalid_reference() {
        let mut proof = SmtProof::new(ProofConclusion::Unsat);
        proof.add_axiom(x_geq_zero());
        // Reference step 5 which does not exist (forward reference from step 1).
        proof.add_resolution(0, 5, "x".into(), smt_false());

        let err = proof.validate();
        assert!(err.is_err());
        let msg = format!("{}", err.unwrap_err());
        assert!(msg.contains("future step") || msg.contains("non-existent"));
    }

    #[test]
    fn test_proof_certificate() {
        let mut proof = SmtProof::new(ProofConclusion::Unsat);
        proof.add_axiom(x_geq_zero());

        let cert = ProofCertificate::new(
            proof,
            vec![0.0, 0.0, 0.0],
            vec![1.0, 2.0, 3.0],
            CertificateVerdict::Verified,
        )
        .with_confidence(0.95)
        .with_linearization_bound(0.01);

        assert!(cert.is_verified());
        assert!((cert.region_volume() - 6.0).abs() < 1e-12);
        assert!((cert.confidence - 0.95).abs() < 1e-12);
        assert!((cert.linearization_error_bound - 0.01).abs() < 1e-12);

        let json = cert.to_json().unwrap();
        assert!(json.contains("Verified"));
    }

    #[test]
    fn test_proof_serialization() {
        let mut proof = SmtProof::new(ProofConclusion::Unsat);
        proof.add_axiom(x_geq_zero());
        proof.add_hypothesis(x_lt_zero());
        proof.add_contradiction(0, 1);

        // JSON round-trip
        let json = ProofSerializer::to_json(&proof).unwrap();
        assert!(json.contains("Unsat"));

        // SMT-LIB2 trace
        let trace = ProofSerializer::to_smtlib2_proof(&proof);
        assert!(trace.contains("axiom"));
        assert!(trace.contains("hypothesis"));
        assert!(trace.contains("contradiction"));

        // Summary
        let summary = ProofSerializer::summary(&proof);
        assert_eq!(summary.num_steps, 3);
        assert_eq!(summary.num_axioms, 1);
        assert_eq!(summary.depth, 2);
        assert_eq!(summary.conclusion, "unsat");
    }

    #[test]
    fn test_proof_builder() {
        let mut builder = ProofBuilder::new(ProofConclusion::Unsat);
        let a = builder.add_axiom(x_geq_zero());
        let h = builder.add_hypothesis(x_lt_zero());
        builder.add_contradiction(a, h);

        let proof = builder.build().unwrap();
        assert_eq!(proof.num_steps(), 3);
        assert_eq!(proof.conclusion, ProofConclusion::Unsat);
    }

    #[test]
    fn test_proof_depth() {
        // Empty proof has depth 0.
        let empty = SmtProof::new(ProofConclusion::Unknown);
        assert_eq!(empty.depth(), 0);

        // Single axiom has depth 1.
        let mut single = SmtProof::new(ProofConclusion::Sat);
        single.add_axiom(x_geq_zero());
        assert_eq!(single.depth(), 1);

        // Chain: axiom -> theory_lemma (leaf) -> resolution(0,1) -> mp(0,2) depth 3.
        let mut proof = SmtProof::new(ProofConclusion::Unsat);
        let a0 = proof.add_axiom(x_geq_zero());
        let a1 = proof.add_theory_lemma("QF_LRA", y_gt_one(), "feasibility");
        let r = proof.add_resolution(a0, a1, "x".into(), smt_false());
        let _mp = proof.add_modus_ponens(a0, r, smt_false());

        // a0 depth=1, a1 depth=1, r depth=2, mp depth=3
        assert_eq!(proof.depth(), 3);
        assert_eq!(proof.leaf_count(), 2);
    }
}
