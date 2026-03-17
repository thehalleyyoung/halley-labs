//! Refinement predicate types and encoding.
//!
//! Provides [`RefinementPredicate`], [`PredicateEncoder`],
//! [`RefinementHistory`], and [`RefinementStrategy`] used by the CEGAR loop
//! to iteratively refine the SMT abstraction.

use crate::{
    ConcreteError, ConcreteResult, SmtExpr, SmtFormula, SmtSort,
};
use crate::{CipherSuite, Extension, HandshakePhase, ProtocolVersion};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;

// ── RefinementPredicate ──────────────────────────────────────────────────

/// A refinement predicate that constrains the SMT search space.
///
/// Generated when a spurious counterexample is detected — the predicate
/// excludes the spurious trace from future solver queries.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RefinementPredicate {
    /// Exclude a specific cipher suite from the adversary's repertoire.
    ExcludeCipher {
        cipher_id: u16,
        reason: String,
    },
    /// Exclude a specific protocol version.
    ExcludeVersion {
        version: ProtocolVersion,
        reason: String,
    },
    /// Require that a specific extension be present in the trace.
    RequireExtension {
        extension_id: u16,
        in_message: String,
        reason: String,
    },
    /// Enforce a message ordering constraint.
    MessageOrdering {
        before: String,
        after: String,
        reason: String,
    },
    /// Add a Dolev-Yao knowledge constraint: the adversary cannot know
    /// a certain value until a specific protocol step.
    KnowledgeConstraint {
        term_name: String,
        not_before_step: usize,
        reason: String,
    },
    /// A raw SMT constraint (fallback for complex refinements).
    RawConstraint {
        variable: String,
        constraint_desc: String,
    },
    /// Exclude a specific combination of cipher + version.
    ExcludeCipherVersionPair {
        cipher_id: u16,
        version: ProtocolVersion,
        reason: String,
    },
    /// Bound the number of adversary actions.
    BoundAdversaryActions {
        max_actions: usize,
        reason: String,
    },
}

impl RefinementPredicate {
    pub fn reason(&self) -> &str {
        match self {
            Self::ExcludeCipher { reason, .. } => reason,
            Self::ExcludeVersion { reason, .. } => reason,
            Self::RequireExtension { reason, .. } => reason,
            Self::MessageOrdering { reason, .. } => reason,
            Self::KnowledgeConstraint { reason, .. } => reason,
            Self::RawConstraint { constraint_desc, .. } => constraint_desc,
            Self::ExcludeCipherVersionPair { reason, .. } => reason,
            Self::BoundAdversaryActions { reason, .. } => reason,
        }
    }

    pub fn predicate_kind(&self) -> &'static str {
        match self {
            Self::ExcludeCipher { .. } => "ExcludeCipher",
            Self::ExcludeVersion { .. } => "ExcludeVersion",
            Self::RequireExtension { .. } => "RequireExtension",
            Self::MessageOrdering { .. } => "MessageOrdering",
            Self::KnowledgeConstraint { .. } => "KnowledgeConstraint",
            Self::RawConstraint { .. } => "RawConstraint",
            Self::ExcludeCipherVersionPair { .. } => "ExcludeCipherVersionPair",
            Self::BoundAdversaryActions { .. } => "BoundAdversaryActions",
        }
    }

    /// Estimate the "strength" of this refinement — stronger predicates
    /// prune more of the search space.
    pub fn strength(&self) -> u32 {
        match self {
            Self::ExcludeCipher { .. } => 2,
            Self::ExcludeVersion { .. } => 5,
            Self::RequireExtension { .. } => 3,
            Self::MessageOrdering { .. } => 4,
            Self::KnowledgeConstraint { .. } => 6,
            Self::RawConstraint { .. } => 1,
            Self::ExcludeCipherVersionPair { .. } => 7,
            Self::BoundAdversaryActions { .. } => 8,
        }
    }
}

impl fmt::Display for RefinementPredicate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ExcludeCipher { cipher_id, reason } => {
                write!(f, "¬cipher(0x{:04x}): {}", cipher_id, reason)
            }
            Self::ExcludeVersion { version, reason } => {
                write!(f, "¬version({}): {}", version, reason)
            }
            Self::RequireExtension { extension_id, in_message, reason } => {
                write!(f, "ext(0x{:04x})∈{}: {}", extension_id, in_message, reason)
            }
            Self::MessageOrdering { before, after, reason } => {
                write!(f, "{} ≺ {}: {}", before, after, reason)
            }
            Self::KnowledgeConstraint { term_name, not_before_step, reason } => {
                write!(f, "K({})≥step({}): {}", term_name, not_before_step, reason)
            }
            Self::RawConstraint { variable, constraint_desc } => {
                write!(f, "raw({}): {}", variable, constraint_desc)
            }
            Self::ExcludeCipherVersionPair { cipher_id, version, reason } => {
                write!(f, "¬(cipher(0x{:04x})∧version({})): {}", cipher_id, version, reason)
            }
            Self::BoundAdversaryActions { max_actions, reason } => {
                write!(f, "|adv_actions|≤{}: {}", max_actions, reason)
            }
        }
    }
}

// ── PredicateEncoder ─────────────────────────────────────────────────────

/// Converts refinement predicates into SMT constraints that can be added
/// to the formula.
pub struct PredicateEncoder {
    /// Variable naming convention for cipher suite variables.
    cipher_var_prefix: String,
    /// Variable naming convention for version variables.
    version_var_prefix: String,
    /// Variable naming convention for extension presence.
    extension_var_prefix: String,
    /// Variable naming convention for message ordering.
    ordering_var_prefix: String,
    /// Variable naming for adversary action count.
    adversary_count_var: String,
    /// Bit-width for bitvector variables.
    bv_width: u32,
}

impl PredicateEncoder {
    pub fn new() -> Self {
        Self {
            cipher_var_prefix: "cipher_selected".into(),
            version_var_prefix: "version_negotiated".into(),
            extension_var_prefix: "ext_present".into(),
            ordering_var_prefix: "msg_time".into(),
            adversary_count_var: "adv_action_count".into(),
            bv_width: 16,
        }
    }

    pub fn with_cipher_var_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.cipher_var_prefix = prefix.into();
        self
    }

    pub fn with_version_var_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.version_var_prefix = prefix.into();
        self
    }

    pub fn with_bv_width(mut self, width: u32) -> Self {
        self.bv_width = width;
        self
    }

    /// Encode a single predicate to an SMT expression.
    pub fn encode_predicate(&self, pred: &RefinementPredicate) -> ConcreteResult<SmtExpr> {
        match pred {
            RefinementPredicate::ExcludeCipher { cipher_id, .. } => {
                Ok(SmtExpr::not(SmtExpr::eq(
                    SmtExpr::var(&self.cipher_var_prefix),
                    SmtExpr::bv_lit(*cipher_id as u64, self.bv_width),
                )))
            }

            RefinementPredicate::ExcludeVersion { version, .. } => {
                let (major, minor) = crate::byte_encoding::version_to_wire(*version);
                let wire_val = ((major as u64) << 8) | minor as u64;
                Ok(SmtExpr::not(SmtExpr::eq(
                    SmtExpr::var(&self.version_var_prefix),
                    SmtExpr::bv_lit(wire_val, self.bv_width),
                )))
            }

            RefinementPredicate::RequireExtension { extension_id, in_message, .. } => {
                let var_name = format!("{}_{}_0x{:04x}", self.extension_var_prefix, in_message, extension_id);
                Ok(SmtExpr::eq(
                    SmtExpr::var(var_name),
                    SmtExpr::BoolLit(true),
                ))
            }

            RefinementPredicate::MessageOrdering { before, after, .. } => {
                let before_var = format!("{}_{}", self.ordering_var_prefix, before);
                let after_var = format!("{}_{}", self.ordering_var_prefix, after);
                Ok(SmtExpr::Lt(
                    Box::new(SmtExpr::var(before_var)),
                    Box::new(SmtExpr::var(after_var)),
                ))
            }

            RefinementPredicate::KnowledgeConstraint { term_name, not_before_step, .. } => {
                let knowledge_var = format!("knowledge_time_{}", term_name);
                Ok(SmtExpr::Le(
                    Box::new(SmtExpr::bv_lit(*not_before_step as u64, self.bv_width)),
                    Box::new(SmtExpr::var(knowledge_var)),
                ))
            }

            RefinementPredicate::RawConstraint { variable, .. } => {
                Ok(SmtExpr::var(variable))
            }

            RefinementPredicate::ExcludeCipherVersionPair { cipher_id, version, .. } => {
                let (major, minor) = crate::byte_encoding::version_to_wire(*version);
                let wire_val = ((major as u64) << 8) | minor as u64;
                Ok(SmtExpr::not(SmtExpr::and(vec![
                    SmtExpr::eq(
                        SmtExpr::var(&self.cipher_var_prefix),
                        SmtExpr::bv_lit(*cipher_id as u64, self.bv_width),
                    ),
                    SmtExpr::eq(
                        SmtExpr::var(&self.version_var_prefix),
                        SmtExpr::bv_lit(wire_val, self.bv_width),
                    ),
                ])))
            }

            RefinementPredicate::BoundAdversaryActions { max_actions, .. } => {
                Ok(SmtExpr::Le(
                    Box::new(SmtExpr::var(&self.adversary_count_var)),
                    Box::new(SmtExpr::bv_lit(*max_actions as u64, self.bv_width)),
                ))
            }
        }
    }

    /// Encode multiple predicates and conjoin them.
    pub fn encode_all(&self, predicates: &[RefinementPredicate]) -> ConcreteResult<SmtExpr> {
        if predicates.is_empty() {
            return Ok(SmtExpr::BoolLit(true));
        }
        let encoded: ConcreteResult<Vec<SmtExpr>> = predicates
            .iter()
            .map(|p| self.encode_predicate(p))
            .collect();
        Ok(SmtExpr::and(encoded?))
    }

    /// Apply refinement predicates to a formula, adding them as assertions.
    pub fn apply_to_formula(
        &self,
        formula: &mut SmtFormula,
        predicates: &[RefinementPredicate],
    ) -> ConcreteResult<usize> {
        let mut count = 0;
        for pred in predicates {
            let expr = self.encode_predicate(pred)?;
            formula.add_assertion(expr);
            count += 1;
        }
        Ok(count)
    }
}

impl Default for PredicateEncoder {
    fn default() -> Self {
        Self::new()
    }
}

// ── RefinementHistory ────────────────────────────────────────────────────

/// Tracks all refinement predicates applied during CEGAR iterations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefinementHistory {
    /// All predicates, indexed by iteration number.
    iterations: Vec<Vec<RefinementPredicate>>,
    /// Set of all unique predicates ever added.
    all_predicates: Vec<RefinementPredicate>,
    /// Count by predicate kind.
    kind_counts: BTreeMap<String, usize>,
}

impl RefinementHistory {
    pub fn new() -> Self {
        Self {
            iterations: Vec::new(),
            all_predicates: Vec::new(),
            kind_counts: BTreeMap::new(),
        }
    }

    /// Record predicates from a single CEGAR iteration.
    pub fn record_iteration(&mut self, predicates: Vec<RefinementPredicate>) {
        for pred in &predicates {
            let kind = pred.predicate_kind().to_string();
            *self.kind_counts.entry(kind).or_insert(0) += 1;
            if !self.all_predicates.contains(pred) {
                self.all_predicates.push(pred.clone());
            }
        }
        self.iterations.push(predicates);
    }

    pub fn iteration_count(&self) -> usize {
        self.iterations.len()
    }

    pub fn total_predicates(&self) -> usize {
        self.all_predicates.len()
    }

    pub fn unique_predicates(&self) -> &[RefinementPredicate] {
        &self.all_predicates
    }

    pub fn predicates_at_iteration(&self, iter: usize) -> Option<&[RefinementPredicate]> {
        self.iterations.get(iter).map(|v| v.as_slice())
    }

    pub fn kind_distribution(&self) -> &BTreeMap<String, usize> {
        &self.kind_counts
    }

    /// Check if adding a predicate would be redundant.
    pub fn is_redundant(&self, pred: &RefinementPredicate) -> bool {
        self.all_predicates.contains(pred)
    }

    /// Compute a minimal subset of predicates that subsumes the full set.
    ///
    /// Uses a greedy algorithm: pick the strongest predicates first,
    /// removing any that are implied by already-selected predicates.
    pub fn minimize(&self) -> Vec<RefinementPredicate> {
        let mut sorted: Vec<&RefinementPredicate> = self.all_predicates.iter().collect();
        // Sort ascending: process broader exclusions (cipher/version) before
        // narrower ones (pairs), so pair redundancy is detected correctly.
        sorted.sort_by(|a, b| a.strength().cmp(&b.strength()));

        let mut minimal: Vec<RefinementPredicate> = Vec::new();
        let mut excluded_ciphers: BTreeSet<u16> = BTreeSet::new();
        let mut excluded_versions: BTreeSet<ProtocolVersion> = BTreeSet::new();
        let mut excluded_pairs: BTreeSet<(u16, ProtocolVersion)> = BTreeSet::new();

        for pred in sorted {
            match pred {
                RefinementPredicate::ExcludeCipher { cipher_id, .. } => {
                    // If we already exclude this cipher via a pair exclusion with
                    // ALL versions, skip. Otherwise add.
                    if excluded_ciphers.insert(*cipher_id) {
                        minimal.push(pred.clone());
                    }
                }
                RefinementPredicate::ExcludeVersion { version, .. } => {
                    if excluded_versions.insert(*version) {
                        minimal.push(pred.clone());
                    }
                }
                RefinementPredicate::ExcludeCipherVersionPair { cipher_id, version, .. } => {
                    // If cipher or version is already fully excluded, skip
                    if excluded_ciphers.contains(cipher_id) || excluded_versions.contains(version) {
                        continue;
                    }
                    if excluded_pairs.insert((*cipher_id, *version)) {
                        minimal.push(pred.clone());
                    }
                }
                _ => {
                    // Other predicates: check simple textual dedup
                    if !minimal.contains(pred) {
                        minimal.push(pred.clone());
                    }
                }
            }
        }

        minimal
    }

    /// Get a summary string.
    pub fn summary(&self) -> String {
        let mut s = format!(
            "RefinementHistory: {} iterations, {} unique predicates\n",
            self.iteration_count(),
            self.total_predicates()
        );
        for (kind, count) in &self.kind_counts {
            s.push_str(&format!("  {}: {}\n", kind, count));
        }
        s
    }
}

impl Default for RefinementHistory {
    fn default() -> Self {
        Self::new()
    }
}

// ── RefinementStrategy ───────────────────────────────────────────────────

/// Strategy for choosing which refinement predicate to add.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RefinementStrategy {
    /// Always pick the strongest predicate available.
    Strongest,
    /// Pick the weakest (most specific) predicate to preserve search space.
    Weakest,
    /// Use interpolation to find a balanced predicate.
    Interpolation,
    /// Cycle through strategies for diversity.
    Adaptive,
}

impl RefinementStrategy {
    /// Select the best predicate from candidates according to this strategy.
    pub fn select<'a>(
        &self,
        candidates: &'a [RefinementPredicate],
        history: &RefinementHistory,
        iteration: usize,
    ) -> Option<&'a RefinementPredicate> {
        if candidates.is_empty() {
            return None;
        }

        match self {
            RefinementStrategy::Strongest => {
                candidates.iter().max_by_key(|p| p.strength())
            }
            RefinementStrategy::Weakest => {
                candidates.iter().min_by_key(|p| p.strength())
            }
            RefinementStrategy::Interpolation => {
                // Prefer predicates that constrain variables not yet constrained.
                let existing_kinds: BTreeSet<&str> = history
                    .unique_predicates()
                    .iter()
                    .map(|p| p.predicate_kind())
                    .collect();
                // Pick a predicate of a kind not yet used, else fall back to strongest
                candidates
                    .iter()
                    .find(|p| !existing_kinds.contains(p.predicate_kind()))
                    .or_else(|| candidates.iter().max_by_key(|p| p.strength()))
            }
            RefinementStrategy::Adaptive => {
                // Alternate between strongest and weakest every other iteration
                if iteration % 2 == 0 {
                    RefinementStrategy::Strongest.select(candidates, history, iteration)
                } else {
                    RefinementStrategy::Weakest.select(candidates, history, iteration)
                }
            }
        }
    }

    /// Select multiple predicates (up to `max_count`).
    pub fn select_multiple<'a>(
        &self,
        candidates: &'a [RefinementPredicate],
        history: &RefinementHistory,
        iteration: usize,
        max_count: usize,
    ) -> Vec<&'a RefinementPredicate> {
        if candidates.is_empty() || max_count == 0 {
            return Vec::new();
        }

        let mut sorted: Vec<&RefinementPredicate> = candidates.iter().collect();
        match self {
            RefinementStrategy::Strongest | RefinementStrategy::Interpolation => {
                sorted.sort_by(|a, b| b.strength().cmp(&a.strength()));
            }
            RefinementStrategy::Weakest => {
                sorted.sort_by(|a, b| a.strength().cmp(&b.strength()));
            }
            RefinementStrategy::Adaptive => {
                if iteration % 2 == 0 {
                    sorted.sort_by(|a, b| b.strength().cmp(&a.strength()));
                } else {
                    sorted.sort_by(|a, b| a.strength().cmp(&b.strength()));
                }
            }
        }

        sorted
            .into_iter()
            .filter(|p| !history.is_redundant(p))
            .take(max_count)
            .collect()
    }
}

impl Default for RefinementStrategy {
    fn default() -> Self {
        Self::Adaptive
    }
}

impl fmt::Display for RefinementStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Strongest => write!(f, "Strongest"),
            Self::Weakest => write!(f, "Weakest"),
            Self::Interpolation => write!(f, "Interpolation"),
            Self::Adaptive => write!(f, "Adaptive"),
        }
    }
}

// ── Interpolation-based refinement ───────────────────────────────────────

/// Attempt to compute a Craig interpolant between the spurious trace
/// constraints and the protocol rules.
///
/// This is a simplified version: in a full implementation this would
/// invoke the SMT solver's interpolation engine.  Here we generate
/// a predicate by analyzing which variables differ between the
/// spurious trace and the protocol constraints.
pub fn interpolation_refine(
    spurious_vars: &BTreeMap<String, u64>,
    protocol_constraints: &[(String, u64, u64)], // (var, min, max)
) -> Vec<RefinementPredicate> {
    let mut predicates = Vec::new();

    for (var, min, max) in protocol_constraints {
        if let Some(&val) = spurious_vars.get(var) {
            if val < *min || val > *max {
                if var.contains("cipher") {
                    predicates.push(RefinementPredicate::ExcludeCipher {
                        cipher_id: val as u16,
                        reason: format!(
                            "interpolation: {} = 0x{:04x} outside [{}, {}]",
                            var, val, min, max
                        ),
                    });
                } else if var.contains("version") {
                    let version = crate::byte_encoding::wire_to_version(
                        (val >> 8) as u8,
                        (val & 0xff) as u8,
                    );
                    predicates.push(RefinementPredicate::ExcludeVersion {
                        version,
                        reason: format!(
                            "interpolation: {} = 0x{:04x} outside [{}, {}]",
                            var, val, min, max
                        ),
                    });
                } else {
                    predicates.push(RefinementPredicate::RawConstraint {
                        variable: var.clone(),
                        constraint_desc: format!(
                            "{} ∈ [{}, {}] (was {})",
                            var, min, max, val
                        ),
                    });
                }
            }
        }
    }

    predicates
}

/// Generate refinement predicates from a comparison of two traces:
/// a spurious abstract trace and a concrete execution.
pub fn diff_based_refinement(
    abstract_ciphers: &[u16],
    abstract_version: ProtocolVersion,
    concrete_valid_ciphers: &BTreeSet<u16>,
    concrete_valid_versions: &BTreeSet<ProtocolVersion>,
) -> Vec<RefinementPredicate> {
    let mut predicates = Vec::new();

    // Exclude ciphers that appeared in the abstract trace but are not valid
    for &cipher in abstract_ciphers {
        if !concrete_valid_ciphers.contains(&cipher) {
            predicates.push(RefinementPredicate::ExcludeCipher {
                cipher_id: cipher,
                reason: format!("cipher 0x{:04x} not in valid set", cipher),
            });
        }
    }

    // Exclude version if not valid
    if !concrete_valid_versions.contains(&abstract_version) {
        predicates.push(RefinementPredicate::ExcludeVersion {
            version: abstract_version,
            reason: format!("{} not in valid version set", abstract_version),
        });
    }

    predicates
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predicate_display() {
        let pred = RefinementPredicate::ExcludeCipher {
            cipher_id: 0x002f,
            reason: "weak cipher".into(),
        };
        let s = format!("{}", pred);
        assert!(s.contains("002f"));
        assert!(s.contains("weak cipher"));
        assert_eq!(pred.predicate_kind(), "ExcludeCipher");
    }

    #[test]
    fn test_predicate_encoder_exclude_cipher() {
        let encoder = PredicateEncoder::new();
        let pred = RefinementPredicate::ExcludeCipher {
            cipher_id: 0xc02f,
            reason: "test".into(),
        };
        let expr = encoder.encode_predicate(&pred).unwrap();
        match expr {
            SmtExpr::Not(_) => {} // Should be a negation
            _ => panic!("Expected Not expression"),
        }
    }

    #[test]
    fn test_predicate_encoder_require_extension() {
        let encoder = PredicateEncoder::new();
        let pred = RefinementPredicate::RequireExtension {
            extension_id: 0x0017,
            in_message: "client_hello".into(),
            reason: "test".into(),
        };
        let expr = encoder.encode_predicate(&pred).unwrap();
        match expr {
            SmtExpr::Eq(_, _) => {}
            _ => panic!("Expected Eq expression"),
        }
    }

    #[test]
    fn test_predicate_encoder_encode_all() {
        let encoder = PredicateEncoder::new();
        let preds = vec![
            RefinementPredicate::ExcludeCipher { cipher_id: 0x002f, reason: "weak".into() },
            RefinementPredicate::ExcludeVersion {
                version: ProtocolVersion::Ssl30,
                reason: "deprecated".into(),
            },
        ];
        let expr = encoder.encode_all(&preds).unwrap();
        match expr {
            SmtExpr::And(exprs) => assert_eq!(exprs.len(), 2),
            _ => panic!("Expected And expression"),
        }
    }

    #[test]
    fn test_predicate_encoder_empty() {
        let encoder = PredicateEncoder::new();
        let expr = encoder.encode_all(&[]).unwrap();
        assert_eq!(expr, SmtExpr::BoolLit(true));
    }

    #[test]
    fn test_refinement_history() {
        let mut history = RefinementHistory::new();
        assert_eq!(history.iteration_count(), 0);

        history.record_iteration(vec![
            RefinementPredicate::ExcludeCipher { cipher_id: 0x002f, reason: "weak".into() },
        ]);
        assert_eq!(history.iteration_count(), 1);
        assert_eq!(history.total_predicates(), 1);

        history.record_iteration(vec![
            RefinementPredicate::ExcludeCipher { cipher_id: 0x002f, reason: "weak".into() },
            RefinementPredicate::ExcludeVersion {
                version: ProtocolVersion::Ssl30,
                reason: "old".into(),
            },
        ]);
        assert_eq!(history.iteration_count(), 2);
        assert_eq!(history.total_predicates(), 2); // deduped
    }

    #[test]
    fn test_refinement_history_minimize() {
        let mut history = RefinementHistory::new();
        history.record_iteration(vec![
            RefinementPredicate::ExcludeCipher { cipher_id: 0x002f, reason: "a".into() },
            RefinementPredicate::ExcludeCipher { cipher_id: 0x002f, reason: "a".into() }, // dup
            RefinementPredicate::ExcludeVersion {
                version: ProtocolVersion::Ssl30,
                reason: "b".into(),
            },
            RefinementPredicate::ExcludeCipherVersionPair {
                cipher_id: 0x002f,
                version: ProtocolVersion::Ssl30,
                reason: "c".into(),
            },
        ]);
        let minimal = history.minimize();
        // The pair exclusion should be dropped because both cipher and version
        // are already fully excluded
        assert!(!minimal.iter().any(|p| matches!(p, RefinementPredicate::ExcludeCipherVersionPair { .. })));
    }

    #[test]
    fn test_refinement_strategy_strongest() {
        let candidates = vec![
            RefinementPredicate::ExcludeCipher { cipher_id: 1, reason: "a".into() },
            RefinementPredicate::BoundAdversaryActions { max_actions: 3, reason: "b".into() },
            RefinementPredicate::RawConstraint { variable: "x".into(), constraint_desc: "c".into() },
        ];
        let history = RefinementHistory::new();
        let selected = RefinementStrategy::Strongest.select(&candidates, &history, 0);
        assert!(selected.is_some());
        assert_eq!(selected.unwrap().predicate_kind(), "BoundAdversaryActions");
    }

    #[test]
    fn test_refinement_strategy_weakest() {
        let candidates = vec![
            RefinementPredicate::ExcludeCipher { cipher_id: 1, reason: "a".into() },
            RefinementPredicate::BoundAdversaryActions { max_actions: 3, reason: "b".into() },
            RefinementPredicate::RawConstraint { variable: "x".into(), constraint_desc: "c".into() },
        ];
        let history = RefinementHistory::new();
        let selected = RefinementStrategy::Weakest.select(&candidates, &history, 0);
        assert!(selected.is_some());
        assert_eq!(selected.unwrap().predicate_kind(), "RawConstraint");
    }

    #[test]
    fn test_refinement_strategy_select_multiple() {
        let candidates = vec![
            RefinementPredicate::ExcludeCipher { cipher_id: 1, reason: "a".into() },
            RefinementPredicate::ExcludeCipher { cipher_id: 2, reason: "b".into() },
            RefinementPredicate::ExcludeVersion { version: ProtocolVersion::Ssl30, reason: "c".into() },
        ];
        let history = RefinementHistory::new();
        let selected = RefinementStrategy::Strongest.select_multiple(&candidates, &history, 0, 2);
        assert_eq!(selected.len(), 2);
    }

    #[test]
    fn test_interpolation_refine() {
        let mut vars = BTreeMap::new();
        vars.insert("cipher_selected".to_string(), 0x002fu64);
        vars.insert("version_negotiated".to_string(), 0x0300u64);

        let constraints = vec![
            ("cipher_selected".to_string(), 0xc000, 0xcfff),
            ("version_negotiated".to_string(), 0x0301, 0x0303),
        ];

        let preds = interpolation_refine(&vars, &constraints);
        assert_eq!(preds.len(), 2);
    }

    #[test]
    fn test_diff_based_refinement() {
        let abstract_ciphers = vec![0x002f, 0xdead];
        let abstract_version = ProtocolVersion::Ssl30;
        let valid_ciphers: BTreeSet<u16> = [0x002f, 0xc02f].iter().copied().collect();
        let valid_versions: BTreeSet<ProtocolVersion> =
            [ProtocolVersion::Tls12, ProtocolVersion::Tls13].iter().copied().collect();

        let preds = diff_based_refinement(
            &abstract_ciphers,
            abstract_version,
            &valid_ciphers,
            &valid_versions,
        );
        // 0xdead not in valid set → ExcludeCipher
        // Ssl30 not in valid set → ExcludeVersion
        assert_eq!(preds.len(), 2);
    }

    #[test]
    fn test_apply_to_formula() {
        let encoder = PredicateEncoder::new();
        let mut formula = SmtFormula::new("test");
        let preds = vec![
            RefinementPredicate::ExcludeCipher { cipher_id: 0x002f, reason: "a".into() },
        ];
        let count = encoder.apply_to_formula(&mut formula, &preds).unwrap();
        assert_eq!(count, 1);
        assert_eq!(formula.assertion_count(), 1);
    }

    #[test]
    fn test_predicate_strength_ordering() {
        let preds = vec![
            RefinementPredicate::RawConstraint { variable: "x".into(), constraint_desc: "c".into() },
            RefinementPredicate::ExcludeCipher { cipher_id: 1, reason: "a".into() },
            RefinementPredicate::BoundAdversaryActions { max_actions: 5, reason: "b".into() },
        ];
        let mut sorted = preds.clone();
        sorted.sort_by(|a, b| b.strength().cmp(&a.strength()));
        assert_eq!(sorted[0].predicate_kind(), "BoundAdversaryActions");
        assert_eq!(sorted[2].predicate_kind(), "RawConstraint");
    }

    #[test]
    fn test_history_summary() {
        let mut history = RefinementHistory::new();
        history.record_iteration(vec![
            RefinementPredicate::ExcludeCipher { cipher_id: 1, reason: "a".into() },
            RefinementPredicate::ExcludeVersion { version: ProtocolVersion::Ssl30, reason: "b".into() },
        ]);
        let summary = history.summary();
        assert!(summary.contains("1 iterations"));
        assert!(summary.contains("2 unique predicates"));
    }
}
