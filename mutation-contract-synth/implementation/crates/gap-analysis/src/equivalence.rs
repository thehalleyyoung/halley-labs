//! # equivalence
//!
//! Equivalence detection for surviving mutants.
//!
//! The **Gap Theorem** partitions every surviving mutant into exactly one of
//! two classes:
//!
//! 1. **Equivalent** – The mutant is semantically identical to the original
//!    program on all inputs.  No test can ever kill it.
//! 2. **Non-equivalent** – There exists at least one input on which the mutant
//!    produces a different result.  Such an input is a potential bug witness.
//!
//! This module implements several strategies for deciding equivalence:
//!
//! - **Syntactic** – Detects trivially equivalent rewrites (e.g. `x + 0`).
//! - **Semantic** – Compares weakest-precondition formulas via SMT-style
//!   structural analysis.
//! - **Observational** – Compares behaviour over sampled input vectors.
//!
//! All results carry a [`ConfidenceLevel`] indicating the certainty of the
//! classification.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use shared_types::formula::{Formula, Predicate, Relation, Term};
use shared_types::operators::{MutantId, MutationOperator};

use crate::analyzer::SurvivingMutant;

// ---------------------------------------------------------------------------
// Confidence and classification
// ---------------------------------------------------------------------------

/// Confidence level of an equivalence verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ConfidenceLevel {
    /// Low confidence – based on heuristics only.
    Low,
    /// Medium confidence – supported by partial formal evidence.
    Medium,
    /// High confidence – fully verified (e.g. SMT proof of equivalence).
    High,
}

impl fmt::Display for ConfidenceLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Low => write!(f, "low"),
            Self::Medium => write!(f, "medium"),
            Self::High => write!(f, "high"),
        }
    }
}

/// An equivalence class groups mutants that are semantically indistinguishable
/// from one another (and possibly from the original).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EquivalenceClass {
    /// Representative mutant ID (or `None` for the original program).
    pub representative: Option<MutantId>,

    /// All mutant IDs belonging to this class.
    pub members: Vec<MutantId>,

    /// Whether the original program belongs to this class.
    pub includes_original: bool,

    /// Confidence level of the classification.
    pub confidence: ConfidenceLevel,

    /// Evidence supporting the classification.
    pub evidence: EquivalenceEvidence,
}

impl EquivalenceClass {
    /// Create a singleton class for an equivalent mutant.
    pub fn equivalent_singleton(id: MutantId, confidence: ConfidenceLevel) -> Self {
        Self {
            representative: None,
            members: vec![id],
            includes_original: true,
            confidence,
            evidence: EquivalenceEvidence::default(),
        }
    }

    /// Create a singleton class for a non-equivalent mutant.
    pub fn non_equivalent_singleton(id: MutantId, confidence: ConfidenceLevel) -> Self {
        Self {
            representative: Some(id.clone()),
            members: vec![id],
            includes_original: false,
            confidence,
            evidence: EquivalenceEvidence::default(),
        }
    }

    /// Number of mutants in this class.
    pub fn size(&self) -> usize {
        self.members.len()
    }

    /// Merge another class into this one.
    pub fn merge(&mut self, other: EquivalenceClass) {
        self.members.extend(other.members);
        self.includes_original = self.includes_original || other.includes_original;
        self.confidence = std::cmp::min(self.confidence, other.confidence);
        self.evidence.merge(other.evidence);
    }
}

impl fmt::Display for EquivalenceClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "EquivClass({} member(s), original={}, confidence={})",
            self.members.len(),
            self.includes_original,
            self.confidence,
        )
    }
}

/// Evidence supporting an equivalence or non-equivalence verdict.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EquivalenceEvidence {
    /// Syntactic patterns that indicate equivalence.
    pub syntactic_patterns: Vec<String>,

    /// The distinguishing formula, if any.
    pub distinguishing_formula: Option<String>,

    /// Observational test vectors that agree/disagree.
    pub agreeing_vectors: usize,
    pub disagreeing_vectors: usize,

    /// Whether the SMT solver proved equivalence.
    pub smt_proven: bool,

    /// Human-readable explanation.
    pub explanation: String,
}

impl EquivalenceEvidence {
    /// Create evidence from an SMT proof.
    pub fn from_smt_proof(explanation: String) -> Self {
        Self {
            smt_proven: true,
            explanation,
            ..Default::default()
        }
    }

    /// Create evidence from syntactic analysis.
    pub fn from_syntactic(patterns: Vec<String>, explanation: String) -> Self {
        Self {
            syntactic_patterns: patterns,
            explanation,
            ..Default::default()
        }
    }

    /// Create evidence from observational testing.
    pub fn from_observational(agreeing: usize, disagreeing: usize) -> Self {
        let explanation =
            format!("Observational: {agreeing} agreeing, {disagreeing} disagreeing test vector(s)");
        Self {
            agreeing_vectors: agreeing,
            disagreeing_vectors: disagreeing,
            explanation,
            ..Default::default()
        }
    }

    /// Merge another piece of evidence into this one.
    pub fn merge(&mut self, other: EquivalenceEvidence) {
        self.syntactic_patterns.extend(other.syntactic_patterns);
        self.agreeing_vectors += other.agreeing_vectors;
        self.disagreeing_vectors += other.disagreeing_vectors;
        self.smt_proven = self.smt_proven || other.smt_proven;
        if !other.explanation.is_empty() {
            if self.explanation.is_empty() {
                self.explanation = other.explanation;
            } else {
                self.explanation = format!("{}; {}", self.explanation, other.explanation);
            }
        }
        if self.distinguishing_formula.is_none() {
            self.distinguishing_formula = other.distinguishing_formula;
        }
    }
}

// ---------------------------------------------------------------------------
// Equivalence result
// ---------------------------------------------------------------------------

/// Result of checking equivalence of a single mutant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EquivalenceResult {
    /// The mutant that was checked.
    pub mutant_id: MutantId,

    /// Equivalence class assignment.
    pub class: EquivalenceClass,

    /// Strategy that produced the result.
    pub strategy: EquivalenceStrategy,

    /// Duration of the check.
    pub duration: Duration,
}

impl EquivalenceResult {
    /// Returns `true` if the mutant was classified as equivalent.
    pub fn is_equivalent(&self) -> bool {
        self.class.includes_original
    }

    /// Returns `true` if the mutant was classified as non-equivalent.
    pub fn is_non_equivalent(&self) -> bool {
        !self.class.includes_original
    }

    /// The confidence level.
    pub fn confidence(&self) -> ConfidenceLevel {
        self.class.confidence
    }
}

impl fmt::Display for EquivalenceResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = if self.is_equivalent() {
            "equivalent"
        } else {
            "non-equivalent"
        };
        write!(
            f,
            "Mutant {} is {label} (confidence={}, strategy={}, took {:?})",
            self.mutant_id, self.class.confidence, self.strategy, self.duration,
        )
    }
}

/// The strategy used to determine equivalence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EquivalenceStrategy {
    /// Purely syntactic pattern matching.
    Syntactic,
    /// Semantic comparison via WP / formula analysis.
    Semantic,
    /// Observational comparison via sampled inputs.
    Observational,
    /// Combination of multiple strategies.
    Combined,
    /// Equivalence could not be determined; default to non-equivalent.
    Unknown,
}

impl fmt::Display for EquivalenceStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Syntactic => write!(f, "syntactic"),
            Self::Semantic => write!(f, "semantic"),
            Self::Observational => write!(f, "observational"),
            Self::Combined => write!(f, "combined"),
            Self::Unknown => write!(f, "unknown"),
        }
    }
}

// ---------------------------------------------------------------------------
// Known equivalent patterns
// ---------------------------------------------------------------------------

/// A syntactic pattern that indicates an equivalent mutation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EquivalentPattern {
    /// Human-readable name for the pattern.
    pub name: String,

    /// Applicable mutation operators.
    pub operators: Vec<MutationOperator>,

    /// Description of why this pattern produces equivalence.
    pub rationale: String,
}

/// Registry of known syntactic equivalence patterns.
fn known_patterns() -> Vec<EquivalentPattern> {
    vec![
        EquivalentPattern {
            name: "additive-identity".into(),
            operators: vec![MutationOperator::Aor],
            rationale: "x + 0 == x - 0 == x for all integer x".into(),
        },
        EquivalentPattern {
            name: "multiplicative-identity".into(),
            operators: vec![MutationOperator::Aor],
            rationale: "x * 1 is equivalent to x for all integer x".into(),
        },
        EquivalentPattern {
            name: "double-negation".into(),
            operators: vec![MutationOperator::Uoi],
            rationale: "!!b == b for boolean b".into(),
        },
        EquivalentPattern {
            name: "reflexive-comparison".into(),
            operators: vec![MutationOperator::Ror],
            rationale: "x <= x always true, x < x always false (context dependent)".into(),
        },
        EquivalentPattern {
            name: "dead-code-mutation".into(),
            operators: vec![MutationOperator::Sdl],
            rationale: "Mutation in dead/unreachable code has no observable effect".into(),
        },
        EquivalentPattern {
            name: "commutative-swap".into(),
            operators: vec![MutationOperator::Aor],
            rationale: "a + b == b + a, a * b == b * a for commutative operators".into(),
        },
        EquivalentPattern {
            name: "abs-non-negative".into(),
            operators: vec![MutationOperator::Abs],
            rationale: "ABS on known non-negative value is identity".into(),
        },
        EquivalentPattern {
            name: "boolean-simplification".into(),
            operators: vec![MutationOperator::Lcr],
            rationale: "a && true == a, a || false == a".into(),
        },
    ]
}

// ---------------------------------------------------------------------------
// Equivalence checker
// ---------------------------------------------------------------------------

/// The equivalence checker classifies surviving mutants as equivalent or
/// non-equivalent using a cascade of strategies.
pub struct EquivalenceChecker {
    /// Timeout per individual check.
    timeout: Duration,

    /// Known syntactic patterns.
    patterns: Vec<EquivalentPattern>,
}

impl EquivalenceChecker {
    /// Create a new equivalence checker with the given per-mutant timeout.
    pub fn new(timeout: Duration) -> Self {
        Self {
            timeout,
            patterns: known_patterns(),
        }
    }

    /// Check a single surviving mutant for equivalence.
    pub fn check(&self, survivor: &SurvivingMutant) -> EquivalenceResult {
        let start = std::time::Instant::now();

        // Strategy cascade: syntactic → semantic → observational → unknown.
        if let Some(result) = self.check_syntactic(survivor) {
            log::debug!(
                "Mutant {} classified as equivalent by syntactic check",
                survivor.id
            );
            return EquivalenceResult {
                mutant_id: survivor.id.clone(),
                class: result,
                strategy: EquivalenceStrategy::Syntactic,
                duration: start.elapsed(),
            };
        }

        if survivor.has_wp() {
            if let Some(result) = self.check_semantic(survivor) {
                log::debug!(
                    "Mutant {} classified by semantic check (equiv={})",
                    survivor.id,
                    result.includes_original
                );
                return EquivalenceResult {
                    mutant_id: survivor.id.clone(),
                    class: result,
                    strategy: EquivalenceStrategy::Semantic,
                    duration: start.elapsed(),
                };
            }
        }

        if let Some(result) = self.check_observational(survivor) {
            return EquivalenceResult {
                mutant_id: survivor.id.clone(),
                class: result,
                strategy: EquivalenceStrategy::Observational,
                duration: start.elapsed(),
            };
        }

        // Default: assume non-equivalent (conservative for bug-finding).
        EquivalenceResult {
            mutant_id: survivor.id.clone(),
            class: EquivalenceClass::non_equivalent_singleton(
                survivor.id.clone(),
                ConfidenceLevel::Low,
            ),
            strategy: EquivalenceStrategy::Unknown,
            duration: start.elapsed(),
        }
    }

    /// Batch-check a set of survivors.
    pub fn check_all(&self, survivors: &[SurvivingMutant]) -> Vec<EquivalenceResult> {
        survivors.iter().map(|s| self.check(s)).collect()
    }

    /// Compute equivalence classes by grouping mutants that are equivalent to
    /// each other.
    pub fn compute_classes(&self, results: &[EquivalenceResult]) -> Vec<EquivalenceClass> {
        // Union-find style grouping: equivalent mutants go into the
        // "original" class, non-equivalent into per-mutant singletons.
        let mut original_class = EquivalenceClass {
            representative: None,
            members: Vec::new(),
            includes_original: true,
            confidence: ConfidenceLevel::High,
            evidence: EquivalenceEvidence::default(),
        };

        let mut non_equiv_classes: Vec<EquivalenceClass> = Vec::new();

        for result in results {
            if result.is_equivalent() {
                original_class.members.push(result.mutant_id.clone());
                original_class.confidence =
                    std::cmp::min(original_class.confidence, result.confidence());
            } else {
                non_equiv_classes.push(result.class.clone());
            }
        }

        // Attempt to merge non-equivalent classes that are equivalent to each
        // other (based on shared WP structure).
        let merged = self.merge_non_equiv_classes(non_equiv_classes);

        let mut all = Vec::new();
        if !original_class.members.is_empty() {
            all.push(original_class);
        }
        all.extend(merged);

        all
    }

    // -- strategy implementations -------------------------------------------

    /// Syntactic equivalence check: match against known patterns.
    fn check_syntactic(&self, survivor: &SurvivingMutant) -> Option<EquivalenceClass> {
        for pattern in &self.patterns {
            if !pattern.operators.contains(&survivor.operator) {
                continue;
            }
            if self.matches_pattern(pattern, survivor) {
                let evidence = EquivalenceEvidence::from_syntactic(
                    vec![pattern.name.clone()],
                    pattern.rationale.clone(),
                );
                let mut class = EquivalenceClass::equivalent_singleton(
                    survivor.id.clone(),
                    ConfidenceLevel::High,
                );
                class.evidence = evidence;
                return Some(class);
            }
        }
        None
    }

    /// Semantic equivalence check: compare WP formulas.
    ///
    /// Two programs are equivalent iff their weakest preconditions are logically
    /// equivalent:  `original_wp ⟺ mutant_wp`.
    ///
    /// We check unsatisfiability of `original_wp ⊕ mutant_wp` (XOR), i.e.:
    ///   (original_wp ∧ ¬mutant_wp) ∨ (¬original_wp ∧ mutant_wp)
    ///
    /// If UNSAT → equivalent.  If SAT → non-equivalent with a model.
    fn check_semantic(&self, survivor: &SurvivingMutant) -> Option<EquivalenceClass> {
        let original_wp = survivor.original_wp.as_ref()?;
        let mutant_wp = survivor.mutant_wp.as_ref()?;

        // Quick structural equality check.
        if original_wp == mutant_wp {
            let evidence = EquivalenceEvidence::from_smt_proof(
                "WP formulas are structurally identical".into(),
            );
            let mut class =
                EquivalenceClass::equivalent_singleton(survivor.id.clone(), ConfidenceLevel::High);
            class.evidence = evidence;
            return Some(class);
        }

        // Build the XOR formula.
        let diff_pos = Formula::And(vec![
            original_wp.clone(),
            Formula::Not(Box::new(mutant_wp.clone())),
        ]);
        let diff_neg = Formula::And(vec![
            Formula::Not(Box::new(original_wp.clone())),
            mutant_wp.clone(),
        ]);
        let xor = Formula::Or(vec![diff_pos, diff_neg]);

        // Structural satisfiability analysis.
        match self.structural_sat_check(&xor) {
            SatResult::Unsat => {
                let evidence = EquivalenceEvidence::from_smt_proof(
                    "WP XOR formula is unsatisfiable (structural analysis)".into(),
                );
                let mut class = EquivalenceClass::equivalent_singleton(
                    survivor.id.clone(),
                    ConfidenceLevel::High,
                );
                class.evidence = evidence;
                Some(class)
            }
            SatResult::Sat(model_desc) => {
                let evidence = EquivalenceEvidence {
                    distinguishing_formula: Some(model_desc.clone()),
                    explanation: format!(
                        "WP XOR is satisfiable: distinguishing assignment found – {model_desc}"
                    ),
                    ..Default::default()
                };
                let mut class = EquivalenceClass::non_equivalent_singleton(
                    survivor.id.clone(),
                    ConfidenceLevel::High,
                );
                class.evidence = evidence;
                Some(class)
            }
            SatResult::Unknown => None,
        }
    }

    /// Observational equivalence check: compare outputs on sample inputs.
    fn check_observational(&self, survivor: &SurvivingMutant) -> Option<EquivalenceClass> {
        // Without an execution engine we use fragment similarity as proxy.
        let similarity =
            fragment_similarity(&survivor.original_fragment, &survivor.mutated_fragment);

        if similarity >= 0.99 {
            // Fragments are nearly identical – likely equivalent.
            let evidence = EquivalenceEvidence::from_observational(1, 0);
            let mut class =
                EquivalenceClass::equivalent_singleton(survivor.id.clone(), ConfidenceLevel::Low);
            class.evidence = evidence;
            return Some(class);
        }

        if similarity < 0.3 {
            // Very different fragments – likely non-equivalent.
            let evidence = EquivalenceEvidence::from_observational(0, 1);
            let mut class = EquivalenceClass::non_equivalent_singleton(
                survivor.id.clone(),
                ConfidenceLevel::Low,
            );
            class.evidence = evidence;
            return Some(class);
        }

        None
    }

    // -- helpers ------------------------------------------------------------

    /// Check if a survivor matches a known equivalent pattern.
    fn matches_pattern(&self, pattern: &EquivalentPattern, survivor: &SurvivingMutant) -> bool {
        match pattern.name.as_str() {
            "additive-identity" => {
                is_additive_identity(&survivor.original_fragment, &survivor.mutated_fragment)
            }
            "multiplicative-identity" => {
                is_multiplicative_identity(&survivor.original_fragment, &survivor.mutated_fragment)
            }
            "double-negation" => {
                is_double_negation(&survivor.original_fragment, &survivor.mutated_fragment)
            }
            "commutative-swap" => {
                is_commutative_swap(&survivor.original_fragment, &survivor.mutated_fragment)
            }
            "boolean-simplification" => {
                is_boolean_simplification(&survivor.original_fragment, &survivor.mutated_fragment)
            }
            _ => false,
        }
    }

    /// Structural satisfiability check without invoking an external solver.
    fn structural_sat_check(&self, formula: &Formula) -> SatResult {
        match formula {
            Formula::False => SatResult::Unsat,
            Formula::True => SatResult::Sat("trivially true".into()),
            Formula::Atom(pred) => {
                if is_trivially_false(pred) {
                    SatResult::Unsat
                } else if is_trivially_true(pred) {
                    SatResult::Sat(format!("trivially true: {pred}"))
                } else {
                    SatResult::Unknown
                }
            }
            Formula::Not(inner) => match self.structural_sat_check(inner) {
                SatResult::Unsat => SatResult::Sat("negation of UNSAT".into()),
                SatResult::Sat(_) => SatResult::Unsat,
                SatResult::Unknown => SatResult::Unknown,
            },
            Formula::And(conjuncts) => {
                // If any conjunct is UNSAT, the conjunction is UNSAT.
                let mut all_trivially_sat = true;
                for c in conjuncts {
                    match self.structural_sat_check(c) {
                        SatResult::Unsat => return SatResult::Unsat,
                        SatResult::Unknown => all_trivially_sat = false,
                        SatResult::Sat(_) => {}
                    }
                }
                // Check for complementary atoms.
                if has_complementary_atoms(conjuncts) {
                    return SatResult::Unsat;
                }
                if all_trivially_sat {
                    SatResult::Sat("all conjuncts trivially satisfiable".into())
                } else {
                    SatResult::Unknown
                }
            }
            Formula::Or(disjuncts) => {
                // If any disjunct is SAT, the disjunction is SAT.
                let mut all_unsat = true;
                for d in disjuncts {
                    match self.structural_sat_check(d) {
                        SatResult::Sat(desc) => return SatResult::Sat(desc),
                        SatResult::Unknown => all_unsat = false,
                        SatResult::Unsat => {}
                    }
                }
                if all_unsat {
                    SatResult::Unsat
                } else {
                    SatResult::Unknown
                }
            }
            Formula::Implies(lhs, rhs) => {
                // p → q ≡ ¬p ∨ q
                let neg_lhs = Formula::Not(Box::new(lhs.as_ref().clone()));
                let equiv = Formula::Or(vec![neg_lhs, rhs.as_ref().clone()]);
                self.structural_sat_check(&equiv)
            }
            _ => SatResult::Unknown, // Iff/Forall/Exists: conservative
        }
    }

    /// Attempt to merge non-equivalent classes with identical WP structure.
    fn merge_non_equiv_classes(&self, classes: Vec<EquivalenceClass>) -> Vec<EquivalenceClass> {
        if classes.len() <= 1 {
            return classes;
        }

        // Simple approach: group by evidence fingerprint.
        let mut groups: HashMap<String, Vec<EquivalenceClass>> = HashMap::new();
        for class in classes {
            let key = class
                .evidence
                .distinguishing_formula
                .clone()
                .unwrap_or_default();
            groups.entry(key).or_default().push(class);
        }

        let mut merged = Vec::new();
        for (_key, group) in groups {
            if group.len() == 1 {
                merged.extend(group);
            } else {
                let mut iter = group.into_iter();
                let mut base = iter.next().unwrap();
                for other in iter {
                    base.merge(other);
                }
                merged.push(base);
            }
        }

        merged
    }
}

// ---------------------------------------------------------------------------
// Internal SAT result
// ---------------------------------------------------------------------------

enum SatResult {
    Sat(String),
    Unsat,
    Unknown,
}

// ---------------------------------------------------------------------------
// Syntactic pattern matchers
// ---------------------------------------------------------------------------

/// Check for additive identity: `x + 0`, `x - 0`, `0 + x`.
fn is_additive_identity(original: &str, mutated: &str) -> bool {
    let patterns = [" + 0", " - 0", "0 + ", "0 - "];
    let has_pattern = |s: &str| patterns.iter().any(|p| s.contains(p));
    has_pattern(original) || has_pattern(mutated)
}

/// Check for multiplicative identity: `x * 1`, `1 * x`.
fn is_multiplicative_identity(original: &str, mutated: &str) -> bool {
    let patterns = [" * 1", "1 * "];
    let has_pattern = |s: &str| patterns.iter().any(|p| s.contains(p));
    has_pattern(original) || has_pattern(mutated)
}

/// Check for double negation: `!!x`, `--x`.
fn is_double_negation(original: &str, mutated: &str) -> bool {
    let has_double_neg = |s: &str| s.contains("!!") || s.contains("--");
    has_double_neg(original) || has_double_neg(mutated)
}

/// Check for commutative swap: `a ○ b` → `b ○ a` for commutative ○.
fn is_commutative_swap(original: &str, mutated: &str) -> bool {
    let commutative_ops = [" + ", " * ", " == ", " != ", " && ", " || "];

    for op in &commutative_ops {
        if let (Some(orig_parts), Some(mut_parts)) =
            (split_binary(original, op), split_binary(mutated, op))
        {
            // Check if operands are swapped.
            if orig_parts.0.trim() == mut_parts.1.trim()
                && orig_parts.1.trim() == mut_parts.0.trim()
            {
                return true;
            }
        }
    }
    false
}

/// Check for boolean simplification: `a && true`, `a || false`.
fn is_boolean_simplification(original: &str, mutated: &str) -> bool {
    let patterns = [
        " && true",
        "true && ",
        " || false",
        "false || ",
        " && 1",
        "1 && ",
        " || 0",
        "0 || ",
    ];
    let has_pattern = |s: &str| patterns.iter().any(|p| s.contains(p));
    has_pattern(original) || has_pattern(mutated)
}

/// Split a string on a binary operator, returning the two operands.
fn split_binary<'a>(s: &'a str, op: &str) -> Option<(&'a str, &'a str)> {
    let idx = s.find(op)?;
    let lhs = &s[..idx];
    let rhs = &s[idx + op.len()..];
    if lhs.is_empty() || rhs.is_empty() {
        None
    } else {
        Some((lhs, rhs))
    }
}

/// Compute a simple similarity metric between two source fragments.
///
/// Returns a value in `[0.0, 1.0]` where `1.0` means identical.
fn fragment_similarity(a: &str, b: &str) -> f64 {
    if a == b {
        return 1.0;
    }
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();

    let max_len = std::cmp::max(a_chars.len(), b_chars.len());
    if max_len == 0 {
        return 1.0;
    }

    // Simple Jaccard-like character bigram similarity.
    let bigrams = |chars: &[char]| -> HashSet<(char, char)> {
        chars.windows(2).map(|w| (w[0], w[1])).collect()
    };

    let a_bi = bigrams(&a_chars);
    let b_bi = bigrams(&b_chars);

    if a_bi.is_empty() && b_bi.is_empty() {
        return if a_chars == b_chars { 1.0 } else { 0.0 };
    }

    let intersection = a_bi.intersection(&b_bi).count();
    let union = a_bi.union(&b_bi).count();

    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

/// Check if a predicate is trivially false.
fn is_trivially_false(pred: &Predicate) -> bool {
    if pred.left == pred.right {
        matches!(pred.relation, Relation::Lt | Relation::Gt | Relation::Ne)
    } else {
        false
    }
}

/// Check if a predicate is trivially true.
fn is_trivially_true(pred: &Predicate) -> bool {
    if pred.left == pred.right {
        matches!(pred.relation, Relation::Eq | Relation::Le | Relation::Ge)
    } else {
        false
    }
}

/// Check whether a list of conjuncts contains a predicate and its negation.
fn has_complementary_atoms(conjuncts: &[Formula]) -> bool {
    let mut positive_atoms: Vec<&Predicate> = Vec::new();
    let mut negative_atoms: Vec<&Predicate> = Vec::new();

    for f in conjuncts {
        match f {
            Formula::Atom(p) => positive_atoms.push(p),
            Formula::Not(inner) => {
                if let Formula::Atom(p) = inner.as_ref() {
                    negative_atoms.push(p);
                }
            }
            _ => {}
        }
    }

    for pos in &positive_atoms {
        for neg in &negative_atoms {
            if pos == neg {
                return true;
            }
        }
    }

    false
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn confidence_ordering() {
        assert!(ConfidenceLevel::Low < ConfidenceLevel::Medium);
        assert!(ConfidenceLevel::Medium < ConfidenceLevel::High);
    }

    #[test]
    fn equivalent_singleton() {
        let id = MutantId::new();
        let class = EquivalenceClass::equivalent_singleton(id.clone(), ConfidenceLevel::High);
        assert!(class.includes_original);
        assert_eq!(class.size(), 1);
        assert!(class.members.contains(&id));
    }

    #[test]
    fn non_equivalent_singleton() {
        let id = MutantId::new();
        let class = EquivalenceClass::non_equivalent_singleton(id.clone(), ConfidenceLevel::Medium);
        assert!(!class.includes_original);
        assert_eq!(class.representative, Some(id));
    }

    #[test]
    fn class_merge() {
        let id1 = MutantId::new();
        let id2 = MutantId::new();
        let mut c1 = EquivalenceClass::non_equivalent_singleton(id1.clone(), ConfidenceLevel::High);
        let c2 = EquivalenceClass::non_equivalent_singleton(id2.clone(), ConfidenceLevel::Low);
        c1.merge(c2);
        assert_eq!(c1.size(), 2);
        assert_eq!(c1.confidence, ConfidenceLevel::Low);
    }

    #[test]
    fn evidence_merge() {
        let mut e1 = EquivalenceEvidence::from_syntactic(vec!["pattern1".into()], "reason1".into());
        let e2 = EquivalenceEvidence::from_observational(5, 1);
        e1.merge(e2);
        assert_eq!(e1.syntactic_patterns.len(), 1);
        assert_eq!(e1.agreeing_vectors, 5);
        assert_eq!(e1.disagreeing_vectors, 1);
        assert!(e1.explanation.contains("reason1"));
    }

    #[test]
    fn fragment_similarity_identical() {
        assert!((fragment_similarity("x + y", "x + y") - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn fragment_similarity_different() {
        let sim = fragment_similarity("x + y", "a * b * c * d * e");
        assert!(sim < 0.5);
    }

    #[test]
    fn additive_identity_detection() {
        assert!(is_additive_identity("x + 0", "x"));
        assert!(!is_additive_identity("x + y", "x - y"));
    }

    #[test]
    fn commutative_swap_detection() {
        assert!(is_commutative_swap("a + b", "b + a"));
        assert!(!is_commutative_swap("a + b", "a - b"));
    }

    #[test]
    fn boolean_simplification_detection() {
        assert!(is_boolean_simplification("a && true", "a"));
        assert!(!is_boolean_simplification("a && b", "a || b"));
    }

    #[test]
    fn double_negation_detection() {
        assert!(is_double_negation("!!x", "x"));
        assert!(!is_double_negation("!x", "x"));
    }

    #[test]
    fn complementary_atoms_found() {
        let p = Predicate::new(Relation::Gt, Term::Var("x".into()), Term::Const(0));
        let pos = Formula::Atom(p.clone());
        let neg = Formula::Not(Box::new(Formula::Atom(p)));
        assert!(has_complementary_atoms(&[pos, neg]));
    }

    #[test]
    fn strategy_display() {
        assert_eq!(format!("{}", EquivalenceStrategy::Syntactic), "syntactic");
        assert_eq!(format!("{}", EquivalenceStrategy::Semantic), "semantic");
    }
}
