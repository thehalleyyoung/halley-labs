//! Security property encoding (Definition D5).
//!
//! Encodes downgrade freedom, version downgrade, extension stripping,
//! and the security ordering on cipher suites. Properties are negated
//! for attack search: ∃ adversary trace violating the property.

use crate::bitvector::{BvEncoder, CipherSuiteEncoder, VersionEncoder};
use crate::unrolling::{TimeStep, UnrollingEngine};
use crate::{
    AdversaryBudget, ConstraintOrigin, EncNegotiationLTS, EncState, Phase, SmtConstraint,
    SmtDeclaration, SmtExpr, SmtSort, StateId,
};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

// ─── Security ordering ──────────────────────────────────────────────────

/// Encodes the total preorder ≤_sec on cipher suites (Definition D5).
#[derive(Debug, Clone)]
pub struct SecurityOrdering {
    /// Map from cipher ID to security score.
    scores: IndexMap<u16, u32>,
    /// Map from version wire value to security level.
    version_levels: IndexMap<u16, u32>,
}

impl SecurityOrdering {
    pub fn new() -> Self {
        SecurityOrdering {
            scores: IndexMap::new(),
            version_levels: IndexMap::new(),
        }
    }

    pub fn with_cipher_scores(scores: IndexMap<u16, u32>) -> Self {
        let mut ordering = Self::new();
        ordering.scores = scores;
        ordering
    }

    /// Register a cipher suite's security score.
    pub fn add_cipher(&mut self, iana_id: u16, score: u32) {
        self.scores.insert(iana_id, score);
    }

    /// Register a version's security level.
    pub fn add_version(&mut self, wire_value: u16, level: u32) {
        self.version_levels.insert(wire_value, level);
    }

    /// Compare two cipher suites: returns true if `a ≤_sec b`.
    pub fn cipher_le(&self, a: u16, b: u16) -> bool {
        let sa = self.scores.get(&a).copied().unwrap_or(0);
        let sb = self.scores.get(&b).copied().unwrap_or(0);
        sa <= sb
    }

    /// Compare two versions: returns true if `a ≤_sec b`.
    pub fn version_le(&self, a: u16, b: u16) -> bool {
        let la = self.version_levels.get(&a).copied().unwrap_or(0);
        let lb = self.version_levels.get(&b).copied().unwrap_or(0);
        la <= lb
    }

    /// Find the strongest cipher in a set.
    pub fn strongest_cipher(&self, set: &BTreeSet<u16>) -> Option<u16> {
        set.iter()
            .max_by_key(|&&id| self.scores.get(&id).copied().unwrap_or(0))
            .copied()
    }

    /// Find the strongest version in a set.
    pub fn strongest_version(&self, set: &BTreeSet<u16>) -> Option<u16> {
        set.iter()
            .max_by_key(|&&v| self.version_levels.get(&v).copied().unwrap_or(0))
            .copied()
    }

    /// Encode the ordering comparison as an SMT expression.
    pub fn encode_cipher_lt(&self, selected: SmtExpr, expected: SmtExpr) -> SmtExpr {
        let sel_score = self.encode_cipher_score(selected);
        let exp_score = self.encode_cipher_score(expected);
        SmtExpr::bv_ult(sel_score, exp_score)
    }

    fn encode_cipher_score(&self, cipher_var: SmtExpr) -> SmtExpr {
        let mut result = SmtExpr::bv_lit(0, 16);
        for (&id, &score) in &self.scores {
            let cond = SmtExpr::eq(cipher_var.clone(), SmtExpr::bv_lit(id as u64, 16));
            result = SmtExpr::ite(cond, SmtExpr::bv_lit(score as u64, 16), result);
        }
        result
    }

    fn encode_version_level(&self, version_var: SmtExpr) -> SmtExpr {
        let mut result = SmtExpr::bv_lit(0, 4);
        for (&ver, &level) in &self.version_levels {
            let cond = SmtExpr::eq(version_var.clone(), SmtExpr::bv_lit(ver as u64, 16));
            result = SmtExpr::ite(cond, SmtExpr::bv_lit(level as u64, 4), result);
        }
        result
    }

    pub fn cipher_scores(&self) -> &IndexMap<u16, u32> {
        &self.scores
    }

    pub fn version_levels(&self) -> &IndexMap<u16, u32> {
        &self.version_levels
    }
}

impl Default for SecurityOrdering {
    fn default() -> Self {
        let mut ordering = Self::new();
        // Default TLS cipher suite ordering
        ordering.add_cipher(0x0000, 0);     // NULL
        ordering.add_cipher(0x0001, 10);    // RSA_NULL_MD5
        ordering.add_cipher(0x002F, 200);   // RSA_AES128_SHA
        ordering.add_cipher(0x0035, 220);   // RSA_AES256_SHA
        ordering.add_cipher(0x009C, 350);   // RSA_AES128_GCM_SHA256
        ordering.add_cipher(0x009D, 370);   // RSA_AES256_GCM_SHA384
        ordering.add_cipher(0xC02F, 450);   // ECDHE_RSA_AES128_GCM
        ordering.add_cipher(0xC030, 470);   // ECDHE_RSA_AES256_GCM
        ordering.add_cipher(0xCCA8, 480);   // ECDHE_RSA_CHACHA20
        ordering.add_cipher(0x1301, 500);   // TLS13_AES128_GCM
        ordering.add_cipher(0x1302, 520);   // TLS13_AES256_GCM
        ordering.add_cipher(0x1303, 510);   // TLS13_CHACHA20

        // Default version ordering
        ordering.add_version(0x0300, 0);  // SSL 3.0
        ordering.add_version(0x0301, 1);  // TLS 1.0
        ordering.add_version(0x0302, 2);  // TLS 1.1
        ordering.add_version(0x0303, 3);  // TLS 1.2
        ordering.add_version(0x0304, 4);  // TLS 1.3

        ordering
    }
}

// ─── Honest outcome computation ─────────────────────────────────────────

/// Computes the expected honest outcome without adversary interference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HonestOutcome {
    pub expected_cipher: Option<u16>,
    pub expected_version: Option<u16>,
    pub expected_extensions: BTreeSet<u16>,
    pub should_succeed: bool,
}

impl HonestOutcome {
    /// Compute the honest outcome from the LTS by finding the path
    /// from initial to terminal with no adversary actions.
    pub fn compute(lts: &EncNegotiationLTS, ordering: &SecurityOrdering) -> Self {
        let mut best_cipher = None;
        let mut best_version = None;
        let mut extensions = BTreeSet::new();
        let mut found_terminal = false;

        // BFS for honest (non-adversary) paths to terminal states
        let mut queue = std::collections::VecDeque::new();
        let mut visited = BTreeSet::new();
        queue.push_back(lts.initial_state);
        visited.insert(lts.initial_state);

        while let Some(state_id) = queue.pop_front() {
            if let Some(state) = lts.states.get(&state_id) {
                if state.is_terminal && state.phase == Phase::Done {
                    found_terminal = true;
                    if let Some(c) = state.selected_cipher {
                        match best_cipher {
                            None => best_cipher = Some(c),
                            Some(prev) => {
                                if ordering.cipher_le(prev, c) {
                                    best_cipher = Some(c);
                                }
                            }
                        }
                    }
                    if let Some(v) = state.selected_version {
                        match best_version {
                            None => best_version = Some(v),
                            Some(prev) => {
                                if ordering.version_le(prev, v) {
                                    best_version = Some(v);
                                }
                            }
                        }
                    }
                    extensions.extend(&state.active_extensions);
                }
            }

            for trans in lts.transitions_from(state_id) {
                if !trans.label.is_adversary() && !visited.contains(&trans.target) {
                    visited.insert(trans.target);
                    queue.push_back(trans.target);
                }
            }
        }

        HonestOutcome {
            expected_cipher: best_cipher,
            expected_version: best_version,
            expected_extensions: extensions,
            should_succeed: found_terminal,
        }
    }

    /// Encode the expected cipher as an SMT expression.
    pub fn cipher_expr(&self) -> Option<SmtExpr> {
        self.expected_cipher
            .map(|c| SmtExpr::bv_lit(c as u64, 16))
    }

    /// Encode the expected version as an SMT expression.
    pub fn version_expr(&self) -> Option<SmtExpr> {
        self.expected_version
            .map(|v| SmtExpr::bv_lit(v as u64, 16))
    }
}

// ─── Downgrade property ─────────────────────────────────────────────────

/// Encodes cipher downgrade freedom (Definition D5).
///
/// A protocol execution exhibits a cipher downgrade if the selected cipher
/// suite is strictly weaker than the strongest mutually supported cipher.
#[derive(Debug, Clone)]
pub struct DowngradeProperty {
    ordering: SecurityOrdering,
    honest_outcome: HonestOutcome,
}

impl DowngradeProperty {
    pub fn new(ordering: SecurityOrdering, honest_outcome: HonestOutcome) -> Self {
        DowngradeProperty {
            ordering,
            honest_outcome,
        }
    }

    /// Encode the downgrade property negation: ∃ execution where
    /// selected_cipher <_sec expected_cipher.
    pub fn encode_negation(&self, final_step: &TimeStep) -> Vec<SmtConstraint> {
        let mut constraints = Vec::new();

        if let Some(expected_cipher) = self.honest_outcome.cipher_expr() {
            // The execution must reach a successful terminal state
            constraints.push(SmtConstraint::new(
                final_step.terminal_expr(),
                ConstraintOrigin::PropertyNegation,
                "property_terminal",
            ));

            // And the selected cipher is strictly weaker than expected
            let downgrade = self.ordering.encode_cipher_lt(
                final_step.cipher_expr(),
                expected_cipher,
            );
            constraints.push(SmtConstraint::new(
                downgrade,
                ConstraintOrigin::PropertyNegation,
                "cipher_downgrade",
            ));

            // The selected cipher must still be valid (not zero/null)
            constraints.push(SmtConstraint::new(
                SmtExpr::not(SmtExpr::eq(
                    final_step.cipher_expr(),
                    SmtExpr::bv_lit(0, 16),
                )),
                ConstraintOrigin::PropertyNegation,
                "cipher_nonzero",
            ));
        }

        constraints
    }

    /// Encode that the final state is a "Done" state (not an abort).
    pub fn encode_successful_completion(
        &self,
        final_step: &TimeStep,
    ) -> SmtConstraint {
        SmtConstraint::new(
            SmtExpr::eq(
                final_step.phase_expr(),
                SmtExpr::bv_lit(Phase::Done.to_index() as u64, 4),
            ),
            ConstraintOrigin::PropertyNegation,
            "successful_completion",
        )
    }

    pub fn ordering(&self) -> &SecurityOrdering {
        &self.ordering
    }

    pub fn honest_outcome(&self) -> &HonestOutcome {
        &self.honest_outcome
    }
}

// ─── Version downgrade property ─────────────────────────────────────────

/// Encodes version downgrade freedom.
#[derive(Debug, Clone)]
pub struct VersionDowngrade {
    ordering: SecurityOrdering,
    honest_outcome: HonestOutcome,
}

impl VersionDowngrade {
    pub fn new(ordering: SecurityOrdering, honest_outcome: HonestOutcome) -> Self {
        VersionDowngrade {
            ordering,
            honest_outcome,
        }
    }

    /// Encode version downgrade negation: ∃ execution where
    /// selected_version <_sec expected_version.
    pub fn encode_negation(&self, final_step: &TimeStep) -> Vec<SmtConstraint> {
        let mut constraints = Vec::new();

        if let Some(expected_version) = self.honest_outcome.version_expr() {
            constraints.push(SmtConstraint::new(
                final_step.terminal_expr(),
                ConstraintOrigin::PropertyNegation,
                "version_property_terminal",
            ));

            let sel_level = self
                .ordering
                .encode_version_level(final_step.version_expr());
            let exp_level = self
                .ordering
                .encode_version_level(expected_version);
            let downgrade = SmtExpr::bv_ult(sel_level, exp_level);

            constraints.push(SmtConstraint::new(
                downgrade,
                ConstraintOrigin::PropertyNegation,
                "version_downgrade",
            ));
        }

        constraints
    }

    pub fn ordering(&self) -> &SecurityOrdering {
        &self.ordering
    }
}

// ─── Extension stripping ────────────────────────────────────────────────

/// Encodes extension stripping attacks.
///
/// An extension stripping attack removes a critical extension
/// from the negotiation that both parties support.
#[derive(Debug, Clone)]
pub struct ExtensionStripping {
    /// Extension IDs that are considered critical.
    critical_extensions: BTreeSet<u16>,
    honest_outcome: HonestOutcome,
}

impl ExtensionStripping {
    pub fn new(critical_extensions: BTreeSet<u16>, honest_outcome: HonestOutcome) -> Self {
        ExtensionStripping {
            critical_extensions,
            honest_outcome,
        }
    }

    /// Standard critical TLS extensions.
    pub fn tls_critical() -> BTreeSet<u16> {
        let mut exts = BTreeSet::new();
        exts.insert(0x002B); // supported_versions
        exts.insert(0x000D); // signature_algorithms
        exts.insert(0x0033); // key_share
        exts.insert(0x002D); // psk_key_exchange_modes
        exts.insert(0x0029); // pre_shared_key
        exts
    }

    /// Encode extension stripping negation: ∃ execution where
    /// a critical extension present in honest outcome is missing.
    pub fn encode_negation(&self, final_step: &TimeStep) -> Vec<SmtConstraint> {
        let mut constraints = Vec::new();

        // Must reach terminal
        constraints.push(SmtConstraint::new(
            final_step.terminal_expr(),
            ConstraintOrigin::PropertyNegation,
            "ext_strip_terminal",
        ));

        // For each critical extension that should be present, encode
        // that it could be absent
        let mut stripped_disjuncts = Vec::new();
        for &ext_id in &self.critical_extensions {
            if self.honest_outcome.expected_extensions.contains(&ext_id) {
                // Extension should be present but is not
                let missing = SmtExpr::not(SmtExpr::select(
                    SmtExpr::var(&final_step.extension_set_var),
                    SmtExpr::bv_lit(ext_id as u64, 16),
                ));
                stripped_disjuncts.push(missing);
            }
        }

        if !stripped_disjuncts.is_empty() {
            constraints.push(SmtConstraint::new(
                SmtExpr::or(stripped_disjuncts),
                ConstraintOrigin::PropertyNegation,
                "extension_stripped",
            ));
        }

        constraints
    }

    pub fn critical_extensions(&self) -> &BTreeSet<u16> {
        &self.critical_extensions
    }
}

// ─── Composite property encoder ─────────────────────────────────────────

/// Property types that can be checked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PropertyKind {
    CipherDowngrade,
    VersionDowngrade,
    ExtensionStripping,
    CipherAndVersion,
    All,
}

/// Composes multiple security properties for encoding.
#[derive(Debug, Clone)]
pub struct PropertyEncoder {
    ordering: SecurityOrdering,
    properties: Vec<PropertyKind>,
    critical_extensions: BTreeSet<u16>,
}

impl PropertyEncoder {
    pub fn new(ordering: SecurityOrdering) -> Self {
        PropertyEncoder {
            ordering,
            properties: vec![PropertyKind::CipherDowngrade],
            critical_extensions: BTreeSet::new(),
        }
    }

    pub fn with_properties(mut self, props: Vec<PropertyKind>) -> Self {
        self.properties = props;
        self
    }

    pub fn with_critical_extensions(mut self, exts: BTreeSet<u16>) -> Self {
        self.critical_extensions = exts;
        self
    }

    /// Encode all configured property negations.
    pub fn encode_property_negation(
        &self,
        lts: &EncNegotiationLTS,
        final_step: &TimeStep,
    ) -> Vec<SmtConstraint> {
        let honest = HonestOutcome::compute(lts, &self.ordering);
        let mut all_constraints = Vec::new();

        // Determine which properties to encode
        let check_cipher = self.properties.iter().any(|p| {
            matches!(
                p,
                PropertyKind::CipherDowngrade | PropertyKind::CipherAndVersion | PropertyKind::All
            )
        });
        let check_version = self.properties.iter().any(|p| {
            matches!(
                p,
                PropertyKind::VersionDowngrade | PropertyKind::CipherAndVersion | PropertyKind::All
            )
        });
        let check_ext = self.properties.iter().any(|p| {
            matches!(p, PropertyKind::ExtensionStripping | PropertyKind::All)
        });

        // Collect property-specific constraints as disjuncts
        // (any one violation is sufficient for an attack)
        let mut violation_disjuncts = Vec::new();

        if check_cipher {
            let prop = DowngradeProperty::new(self.ordering.clone(), honest.clone());
            let constraints = prop.encode_negation(final_step);
            if constraints.len() > 1 {
                let cipher_violation: Vec<SmtExpr> =
                    constraints.iter().map(|c| c.formula.clone()).collect();
                violation_disjuncts.push(SmtExpr::and(cipher_violation));
            }
        }

        if check_version {
            let prop = VersionDowngrade::new(self.ordering.clone(), honest.clone());
            let constraints = prop.encode_negation(final_step);
            if constraints.len() > 1 {
                let version_violation: Vec<SmtExpr> =
                    constraints.iter().map(|c| c.formula.clone()).collect();
                violation_disjuncts.push(SmtExpr::and(version_violation));
            }
        }

        if check_ext && !self.critical_extensions.is_empty() {
            let prop = ExtensionStripping::new(
                self.critical_extensions.clone(),
                honest.clone(),
            );
            let constraints = prop.encode_negation(final_step);
            if constraints.len() > 1 {
                let ext_violation: Vec<SmtExpr> =
                    constraints.iter().map(|c| c.formula.clone()).collect();
                violation_disjuncts.push(SmtExpr::and(ext_violation));
            }
        }

        // Must reach terminal state
        all_constraints.push(SmtConstraint::new(
            final_step.terminal_expr(),
            ConstraintOrigin::PropertyNegation,
            "property_requires_terminal",
        ));

        // Must be a successful completion (not abort)
        all_constraints.push(SmtConstraint::new(
            SmtExpr::eq(
                final_step.phase_expr(),
                SmtExpr::bv_lit(Phase::Done.to_index() as u64, 4),
            ),
            ConstraintOrigin::PropertyNegation,
            "property_requires_success",
        ));

        // At least one property is violated
        if !violation_disjuncts.is_empty() {
            all_constraints.push(SmtConstraint::new(
                SmtExpr::or(violation_disjuncts),
                ConstraintOrigin::PropertyNegation,
                "some_property_violated",
            ));
        }

        all_constraints
    }

    pub fn ordering(&self) -> &SecurityOrdering {
        &self.ordering
    }

    pub fn properties(&self) -> &[PropertyKind] {
        &self.properties
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AdvAction, EncState, EncTransition, EncTransitionLabel};

    fn make_test_lts() -> EncNegotiationLTS {
        let init = EncState::new(StateId(0), Phase::Init);
        let mut lts = EncNegotiationLTS::new(init);

        let mut sh = EncState::new(StateId(1), Phase::ServerHelloReceived);
        sh.selected_cipher = Some(0x1302);
        sh.selected_version = Some(0x0304);
        sh.active_extensions.insert(0x002B);
        lts.add_state(sh);

        let mut done = EncState::new(StateId(2), Phase::Done);
        done.selected_cipher = Some(0x1302);
        done.selected_version = Some(0x0304);
        done.active_extensions.insert(0x002B);
        lts.add_state(done);

        lts.add_transition(EncTransition {
            source: StateId(0),
            target: StateId(1),
            label: EncTransitionLabel::Tau,
            guard: None,
        });
        lts.add_transition(EncTransition {
            source: StateId(1),
            target: StateId(2),
            label: EncTransitionLabel::Tau,
            guard: None,
        });

        lts
    }

    #[test]
    fn test_security_ordering_default() {
        let ordering = SecurityOrdering::default();
        assert!(ordering.cipher_le(0x002F, 0x1302));
        assert!(!ordering.cipher_le(0x1302, 0x002F));
        assert!(ordering.version_le(0x0301, 0x0304));
    }

    #[test]
    fn test_security_ordering_strongest() {
        let ordering = SecurityOrdering::default();
        let mut set = BTreeSet::new();
        set.insert(0x002F);
        set.insert(0x1302);
        let strongest = ordering.strongest_cipher(&set);
        assert_eq!(strongest, Some(0x1302));
    }

    #[test]
    fn test_honest_outcome() {
        let lts = make_test_lts();
        let ordering = SecurityOrdering::default();
        let outcome = HonestOutcome::compute(&lts, &ordering);
        assert!(outcome.should_succeed);
        assert_eq!(outcome.expected_cipher, Some(0x1302));
        assert_eq!(outcome.expected_version, Some(0x0304));
    }

    #[test]
    fn test_downgrade_property_negation() {
        let lts = make_test_lts();
        let ordering = SecurityOrdering::default();
        let honest = HonestOutcome::compute(&lts, &ordering);
        let prop = DowngradeProperty::new(ordering, honest);

        let step = TimeStep::new(5);
        let constraints = prop.encode_negation(&step);
        assert!(!constraints.is_empty());

        let labels: Vec<&str> = constraints.iter().map(|c| c.label.as_str()).collect();
        assert!(labels.contains(&"cipher_downgrade"));
    }

    #[test]
    fn test_version_downgrade_negation() {
        let lts = make_test_lts();
        let ordering = SecurityOrdering::default();
        let honest = HonestOutcome::compute(&lts, &ordering);
        let prop = VersionDowngrade::new(ordering, honest);

        let step = TimeStep::new(5);
        let constraints = prop.encode_negation(&step);
        assert!(!constraints.is_empty());
    }

    #[test]
    fn test_extension_stripping() {
        let lts = make_test_lts();
        let ordering = SecurityOrdering::default();
        let honest = HonestOutcome::compute(&lts, &ordering);
        let critical = ExtensionStripping::tls_critical();
        let prop = ExtensionStripping::new(critical, honest);

        let step = TimeStep::new(5);
        let constraints = prop.encode_negation(&step);
        assert!(!constraints.is_empty());
    }

    #[test]
    fn test_property_encoder_cipher() {
        let lts = make_test_lts();
        let ordering = SecurityOrdering::default();
        let encoder = PropertyEncoder::new(ordering)
            .with_properties(vec![PropertyKind::CipherDowngrade]);

        let step = TimeStep::new(5);
        let constraints = encoder.encode_property_negation(&lts, &step);
        assert!(!constraints.is_empty());

        let has_terminal = constraints.iter().any(|c| c.label.contains("terminal"));
        assert!(has_terminal);
    }

    #[test]
    fn test_property_encoder_all() {
        let lts = make_test_lts();
        let ordering = SecurityOrdering::default();
        let critical = ExtensionStripping::tls_critical();
        let encoder = PropertyEncoder::new(ordering)
            .with_properties(vec![PropertyKind::All])
            .with_critical_extensions(critical);

        let step = TimeStep::new(10);
        let constraints = encoder.encode_property_negation(&lts, &step);
        assert!(constraints.len() >= 3);
    }

    #[test]
    fn test_security_ordering_encode() {
        let ordering = SecurityOrdering::default();
        let lt = ordering.encode_cipher_lt(SmtExpr::var("sel"), SmtExpr::var("exp"));
        let s = format!("{}", lt);
        assert!(s.contains("bvult"));
    }

    #[test]
    fn test_honest_outcome_no_done() {
        let init = EncState::new(StateId(0), Phase::Init);
        let lts = EncNegotiationLTS::new(init);
        let ordering = SecurityOrdering::default();
        let outcome = HonestOutcome::compute(&lts, &ordering);
        assert!(!outcome.should_succeed);
        assert!(outcome.expected_cipher.is_none());
    }

    #[test]
    fn test_downgrade_successful_completion() {
        let ordering = SecurityOrdering::default();
        let honest = HonestOutcome {
            expected_cipher: Some(0x1302),
            expected_version: Some(0x0304),
            expected_extensions: BTreeSet::new(),
            should_succeed: true,
        };
        let prop = DowngradeProperty::new(ordering, honest);
        let step = TimeStep::new(5);
        let constraint = prop.encode_successful_completion(&step);
        let s = format!("{}", constraint.formula);
        assert!(s.contains("phase_t5"));
    }

    #[test]
    fn test_extension_stripping_critical() {
        let critical = ExtensionStripping::tls_critical();
        assert!(critical.contains(&0x002B)); // supported_versions
        assert!(critical.contains(&0x000D)); // signature_algorithms
    }

    #[test]
    fn test_security_ordering_custom() {
        let mut scores = IndexMap::new();
        scores.insert(0x1111, 100);
        scores.insert(0x2222, 200);
        let ordering = SecurityOrdering::with_cipher_scores(scores);
        assert!(ordering.cipher_le(0x1111, 0x2222));
        assert!(!ordering.cipher_le(0x2222, 0x1111));
    }

    #[test]
    fn test_property_kind_variants() {
        let kinds = vec![
            PropertyKind::CipherDowngrade,
            PropertyKind::VersionDowngrade,
            PropertyKind::ExtensionStripping,
            PropertyKind::CipherAndVersion,
            PropertyKind::All,
        ];
        assert_eq!(kinds.len(), 5);
    }
}
