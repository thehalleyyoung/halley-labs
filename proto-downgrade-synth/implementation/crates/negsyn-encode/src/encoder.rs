//! Core DY+SMT encoding (ALG4: DYENCODE).
//!
//! The `DYEncoder` orchestrates the complete encoding pipeline:
//! 1. Initialize bitvector encoders for cipher suites and versions
//! 2. Unroll the LTS transitions up to depth k
//! 3. Encode Dolev-Yao adversary knowledge and deduction
//! 4. Encode adversary actions with budget constraints
//! 5. Encode security property negation
//! 6. Apply optimizations
//! 7. Generate SMT-LIB2 output

use crate::adversary_encoding::{AdversaryEncoder, AdversaryEncoderConfig};
use crate::bitvector::BvEncoder;
use crate::dolev_yao::{DYTerm, DYTermAlgebra, DeductionRules, KnowledgeEncoder, TermEncoder};
use crate::optimization::{EncodingOptimizer, OptimizationConfig, OptimizationStats};
use crate::property::{HonestOutcome, PropertyEncoder, PropertyKind, SecurityOrdering};
use crate::smtlib::{SmtLib2Writer, WriterConfig};
use crate::unrolling::{UnrollingConfig, UnrollingEngine};
use crate::{
    AdversaryBudget, ConstraintOrigin, EncNegotiationLTS, EncState, EncTransition,
    EncTransitionLabel, EncodingStats, Phase, SmtConstraint, SmtDeclaration, SmtExpr, SmtFormula,
    SmtSort, StateId,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::time::Instant;

// ─── Encoder configuration ──────────────────────────────────────────────

/// Configuration for the DYEncoder.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncoderConfig {
    pub max_depth: u32,
    pub adversary_budget: AdversaryBudget,
    pub properties: Vec<PropertyKind>,
    pub critical_extensions: BTreeSet<u16>,
    pub optimization: OptimizationConfig,
    pub adversary_config: AdversaryEncoderConfig,
    pub unrolling_config: UnrollingConfig,
    pub writer_config: WriterConfig,
    pub max_derivation_depth: u32,
    pub include_dy_algebra: bool,
    pub include_knowledge_encoding: bool,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        EncoderConfig {
            max_depth: 20,
            adversary_budget: AdversaryBudget::new(5),
            properties: vec![PropertyKind::CipherDowngrade],
            critical_extensions: BTreeSet::new(),
            optimization: OptimizationConfig::default(),
            adversary_config: AdversaryEncoderConfig::default(),
            unrolling_config: UnrollingConfig::default(),
            writer_config: WriterConfig::default(),
            max_derivation_depth: 3,
            include_dy_algebra: true,
            include_knowledge_encoding: true,
        }
    }
}

impl EncoderConfig {
    pub fn with_depth(mut self, depth: u32) -> Self {
        self.max_depth = depth;
        self.unrolling_config.max_depth = depth;
        self
    }

    pub fn with_budget(mut self, budget: u32) -> Self {
        self.adversary_budget = AdversaryBudget::new(budget);
        self
    }

    pub fn with_properties(mut self, props: Vec<PropertyKind>) -> Self {
        self.properties = props;
        self
    }

    pub fn with_critical_extensions(mut self, exts: BTreeSet<u16>) -> Self {
        self.critical_extensions = exts;
        self
    }
}

// ─── Encoding result ────────────────────────────────────────────────────

/// The complete result of the DYENCODE algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DyEncodeResult {
    pub formula: SmtFormula,
    pub stats: EncodingStats,
    pub optimization_stats: OptimizationStats,
    pub smtlib2: String,
}

impl DyEncodeResult {
    pub fn is_trivially_unsat(&self) -> bool {
        self.formula
            .constraints
            .iter()
            .any(|c| matches!(&c.formula, SmtExpr::BoolLit(false)))
    }

    pub fn constraint_count(&self) -> usize {
        self.formula.constraint_count()
    }

    pub fn declaration_count(&self) -> usize {
        self.formula.declaration_count()
    }
}

// ─── DYEncoder ──────────────────────────────────────────────────────────

/// Main encoder orchestrating the complete ALG4: DYENCODE pipeline.
///
/// Takes a NegotiationLTS and AdversaryBudget and produces an SMT formula
/// whose satisfiability witnesses a downgrade attack.
#[derive(Debug)]
pub struct DYEncoder {
    config: EncoderConfig,
    bv_encoder: BvEncoder,
    unrolling: UnrollingEngine,
    adversary: AdversaryEncoder,
    knowledge: KnowledgeEncoder,
    term_algebra: DYTermAlgebra,
    property_encoder: PropertyEncoder,
    optimizer: EncodingOptimizer,
}

impl DYEncoder {
    pub fn new(config: EncoderConfig) -> Self {
        let bv_encoder = BvEncoder::new(&[]);
        let unrolling = UnrollingEngine::new(config.unrolling_config.clone());
        let adversary = AdversaryEncoder::new(
            config.adversary_budget.clone(),
            config.adversary_config.clone(),
        );
        let knowledge = KnowledgeEncoder::new(config.max_derivation_depth);
        let term_algebra = DYTermAlgebra::new();
        let ordering = SecurityOrdering::default();
        let property_encoder = PropertyEncoder::new(ordering)
            .with_properties(config.properties.clone())
            .with_critical_extensions(config.critical_extensions.clone());
        let optimizer = EncodingOptimizer::new(config.optimization.clone());

        DYEncoder {
            config,
            bv_encoder,
            unrolling,
            adversary,
            knowledge,
            term_algebra,
            property_encoder,
            optimizer,
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(EncoderConfig::default())
    }

    /// Main encoding method: produces the complete SMT formula.
    ///
    /// Implements Algorithm 4 (DYENCODE) from the formal specification:
    /// 1. State encoding: bitvector variables for each time step
    /// 2. Transition encoding: LTS transitions as SMT constraints
    /// 3. Adversary encoding: bounded DY actions
    /// 4. Knowledge encoding: DY deduction rules
    /// 5. Property encoding: downgrade freedom negation
    /// 6. Optimization: simplify and reduce the formula
    pub fn encode_lts(
        &mut self,
        lts: &EncNegotiationLTS,
        budget: &AdversaryBudget,
    ) -> DyEncodeResult {
        let start = Instant::now();

        // Phase 0: Initialize sub-encoders from the LTS
        let cipher_ids: Vec<u16> = lts.all_cipher_ids().into_iter().collect();
        self.bv_encoder = BvEncoder::new(&cipher_ids);
        self.unrolling = UnrollingEngine::new(UnrollingConfig {
            max_depth: self.config.max_depth,
            ..self.config.unrolling_config.clone()
        });
        self.unrolling.initialize(lts);
        self.adversary = AdversaryEncoder::new(budget.clone(), self.config.adversary_config.clone());
        self.adversary.initialize(self.config.max_depth);

        let mut formula = SmtFormula::new(self.config.max_depth, budget.max_actions);
        formula.library_name = "negsyn-encode".to_string();
        formula.library_version = "0.1.0".to_string();

        // Phase 1: Declarations
        self.emit_declarations(&mut formula, lts);

        // Phase 2: Initial state constraints
        self.emit_initial_state(&mut formula, lts);

        // Phase 3: Transition unrolling
        self.emit_transitions(&mut formula, lts);

        // Phase 4: Adversary action constraints
        self.emit_adversary_constraints(&mut formula);

        // Phase 5: DY knowledge encoding
        if self.config.include_knowledge_encoding {
            self.emit_knowledge_constraints(&mut formula, lts);
        }

        // Phase 6: DY term algebra
        if self.config.include_dy_algebra {
            self.emit_dy_algebra(&mut formula);
        }

        // Phase 7: Property negation
        self.emit_property_negation(&mut formula, lts);

        // Phase 8: Termination requirement
        self.emit_termination(&mut formula);

        let encoding_time = start.elapsed().as_millis() as u64;

        // Phase 9: Optimization
        let opt_start = Instant::now();
        self.optimizer.optimize(&mut formula);
        let optimization_time = opt_start.elapsed().as_millis() as u64;

        // Phase 10: Generate SMT-LIB2 output
        let writer = SmtLib2Writer::new(self.config.writer_config.clone());
        let smtlib2 = writer.write_formula(&formula);

        let stats = EncodingStats {
            total_constraints: formula.constraint_count(),
            total_variables: formula.declaration_count(),
            total_nodes: formula.total_nodes(),
            unrolling_depth: self.config.max_depth,
            adversary_budget: budget.max_actions,
            encoding_time_ms: encoding_time,
            optimization_time_ms: optimization_time,
            state_vars: lts.state_count(),
            transition_constraints: lts.transition_count(),
            knowledge_constraints: 0,
            property_constraints: formula
                .constraints
                .iter()
                .filter(|c| matches!(c.origin, ConstraintOrigin::PropertyNegation))
                .count(),
        };

        DyEncodeResult {
            formula,
            stats,
            optimization_stats: self.optimizer.stats().clone(),
            smtlib2,
        }
    }

    /// Emit all variable declarations.
    fn emit_declarations(&self, formula: &mut SmtFormula, lts: &EncNegotiationLTS) {
        // Unrolling state declarations
        for decl in self.unrolling.declarations() {
            formula.add_declaration(decl);
        }

        // Adversary action declarations
        for decl in self.adversary.declarations() {
            formula.add_declaration(decl);
        }

        // Bitvector axiom function declarations
        for decl in self.bv_encoder.cipher_encoder.declarations() {
            formula.add_declaration(decl);
        }
    }

    /// Emit initial state constraints.
    fn emit_initial_state(&self, formula: &mut SmtFormula, lts: &EncNegotiationLTS) {
        for c in self.unrolling.encode_initial_state(lts) {
            formula.add_constraint(c);
        }

        // Bitvector axioms (score definitions)
        for axiom in self.bv_encoder.base_axioms() {
            formula.add_constraint(SmtConstraint::new(
                axiom,
                ConstraintOrigin::InitialState,
                "cipher_score_axiom",
            ));
        }
    }

    /// Emit transition unrolling constraints.
    fn emit_transitions(&self, formula: &mut SmtFormula, lts: &EncNegotiationLTS) {
        for c in self.unrolling.encode_all_transitions(lts) {
            formula.add_constraint(c);
        }

        // Valid state constraints
        for c in self.unrolling.encode_valid_states(lts) {
            formula.add_constraint(c);
        }
    }

    /// Emit adversary budget and action constraints.
    fn emit_adversary_constraints(&self, formula: &mut SmtFormula) {
        for c in self.adversary.encode_all(self.config.max_depth) {
            formula.add_constraint(c);
        }
    }

    /// Emit Dolev-Yao knowledge encoding.
    fn emit_knowledge_constraints(
        &self,
        formula: &mut SmtFormula,
        lts: &EncNegotiationLTS,
    ) {
        // Build the set of relevant DY terms from the LTS
        let terms = self.extract_terms_from_lts(lts);

        if terms.is_empty() {
            return;
        }

        // Declare knowledge variables for each step
        for step in 0..=self.config.max_depth {
            for decl in self.knowledge.declare_knowledge_vars(&terms, step) {
                formula.add_declaration(decl);
            }
        }

        // Public terms (cipher suite IDs, version IDs are public)
        let public_terms: Vec<DYTerm> = terms
            .iter()
            .filter(|t| {
                matches!(
                    t,
                    DYTerm::CipherSuiteId(_) | DYTerm::VersionId(_) | DYTerm::ExtensionId(_)
                )
            })
            .cloned()
            .collect();

        // Initial knowledge
        for formula_expr in self.knowledge.encode_initial_knowledge(&public_terms, &terms) {
            formula.add_constraint(SmtConstraint::new(
                formula_expr,
                ConstraintOrigin::KnowledgeAccumulation { step: 0 },
                "initial_knowledge",
            ));
        }

        // Knowledge monotonicity at each step
        for step in 0..self.config.max_depth {
            for mono in self.knowledge.encode_monotonicity(&terms, step) {
                formula.add_constraint(SmtConstraint::new(
                    mono,
                    ConstraintOrigin::KnowledgeAccumulation { step },
                    format!("knowledge_monotonicity_step_{}", step),
                ));
            }
        }

        // Deduction rules at each step
        let deduction_rules = DeductionRules::new(terms.clone());
        for step in 0..=self.config.max_depth {
            for c in deduction_rules.encode_closure(&self.knowledge, step) {
                formula.add_constraint(c);
            }
        }
    }

    /// Emit DY term algebra declarations and axioms.
    fn emit_dy_algebra(&mut self, formula: &mut SmtFormula) {
        for decl in self.term_algebra.sort_declarations() {
            formula.add_declaration(decl);
        }
        for decl in self.term_algebra.constructor_declarations() {
            formula.add_declaration(decl);
        }
        for axiom in self.term_algebra.constructor_destructor_axioms() {
            formula.add_constraint(SmtConstraint::new(
                axiom,
                ConstraintOrigin::InitialState,
                "dy_axiom",
            ));
        }
    }

    /// Emit security property negation constraints.
    fn emit_property_negation(
        &self,
        formula: &mut SmtFormula,
        lts: &EncNegotiationLTS,
    ) {
        let final_step = match self.unrolling.time_step(self.config.max_depth) {
            Some(ts) => ts,
            None => return,
        };

        for c in self.property_encoder.encode_property_negation(lts, final_step) {
            formula.add_constraint(c);
        }
    }

    /// Emit termination requirement.
    fn emit_termination(&self, formula: &mut SmtFormula) {
        formula.add_constraint(self.unrolling.encode_termination_requirement());
    }

    /// Extract relevant DY terms from the LTS structure.
    fn extract_terms_from_lts(&self, lts: &EncNegotiationLTS) -> Vec<DYTerm> {
        let mut terms = Vec::new();
        let mut seen = BTreeSet::new();

        // Cipher suite IDs as DY terms
        for &id in &lts.all_cipher_ids() {
            let term = DYTerm::CipherSuiteId(id);
            if seen.insert(term.clone()) {
                terms.push(term);
            }
        }

        // Version IDs as DY terms
        for &ver in &lts.all_version_ids() {
            let term = DYTerm::VersionId(ver);
            if seen.insert(term.clone()) {
                terms.push(term);
            }
        }

        // Extension IDs
        for state in lts.states.values() {
            for &ext_id in &state.active_extensions {
                let term = DYTerm::ExtensionId(ext_id);
                if seen.insert(term.clone()) {
                    terms.push(term);
                }
            }
        }

        // Add nonces for randomness
        let nonce_0 = DYTerm::nonce(0);
        if seen.insert(nonce_0.clone()) {
            terms.push(nonce_0);
        }
        let nonce_1 = DYTerm::nonce(1);
        if seen.insert(nonce_1.clone()) {
            terms.push(nonce_1);
        }

        terms
    }

    /// Perform incremental encoding: add constraints for a new depth increment.
    pub fn encode_incremental(
        &mut self,
        lts: &EncNegotiationLTS,
        previous_formula: &SmtFormula,
        new_depth: u32,
    ) -> SmtFormula {
        let old_depth = previous_formula.depth_bound;
        let mut formula = previous_formula.clone();
        formula.depth_bound = new_depth;

        // Add declarations for new time steps
        for step in (old_depth + 1)..=new_depth {
            let ts = crate::unrolling::TimeStep::new(step);
            for decl in ts.declarations(self.unrolling.state_bv_width()) {
                formula.add_declaration(decl);
            }
        }

        // Add transitions for new steps
        for step in old_depth..new_depth {
            for c in self.unrolling.encode_transitions_at_step(lts, step) {
                formula.add_constraint(c);
            }
        }

        // Update termination requirement
        if let Some(final_step) = self.unrolling.time_step(new_depth) {
            formula.add_constraint(SmtConstraint::new(
                final_step.terminal_expr(),
                ConstraintOrigin::DepthBound,
                "must_terminate_incremental",
            ));
        }

        formula
    }

    pub fn config(&self) -> &EncoderConfig {
        &self.config
    }

    pub fn unrolling(&self) -> &UnrollingEngine {
        &self.unrolling
    }

    pub fn adversary_encoder(&self) -> &AdversaryEncoder {
        &self.adversary
    }

    pub fn bv_encoder(&self) -> &BvEncoder {
        &self.bv_encoder
    }

    pub fn property_encoder(&self) -> &PropertyEncoder {
        &self.property_encoder
    }
}

// ─── Convenience functions ──────────────────────────────────────────────

/// Quick encode with default settings.
pub fn encode_lts_default(lts: &EncNegotiationLTS) -> DyEncodeResult {
    let config = EncoderConfig::default();
    let budget = config.adversary_budget.clone();
    let mut encoder = DYEncoder::new(config);
    encoder.encode_lts(lts, &budget)
}

/// Encode with specific depth and budget.
pub fn encode_lts(
    lts: &EncNegotiationLTS,
    depth: u32,
    budget: u32,
) -> DyEncodeResult {
    let config = EncoderConfig::default().with_depth(depth).with_budget(budget);
    let adv_budget = config.adversary_budget.clone();
    let mut encoder = DYEncoder::new(config);
    encoder.encode_lts(lts, &adv_budget)
}

/// Encode checking all property types.
pub fn encode_lts_all_properties(
    lts: &EncNegotiationLTS,
    depth: u32,
    budget: u32,
) -> DyEncodeResult {
    let config = EncoderConfig::default()
        .with_depth(depth)
        .with_budget(budget)
        .with_properties(vec![PropertyKind::All])
        .with_critical_extensions(crate::property::ExtensionStripping::tls_critical());
    let adv_budget = config.adversary_budget.clone();
    let mut encoder = DYEncoder::new(config);
    encoder.encode_lts(lts, &adv_budget)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AdvAction, EncState, EncTransition, EncTransitionLabel};

    fn make_simple_lts() -> EncNegotiationLTS {
        let init = EncState::new(StateId(0), Phase::Init);
        let mut lts = EncNegotiationLTS::new(init);

        let mut ch = EncState::new(StateId(1), Phase::ClientHelloSent);
        ch.offered_ciphers.insert(0x002F);
        ch.offered_ciphers.insert(0x009C);
        ch.offered_ciphers.insert(0x1302);
        lts.add_state(ch);

        let mut sh = EncState::new(StateId(2), Phase::ServerHelloReceived);
        sh.selected_cipher = Some(0x1302);
        sh.selected_version = Some(0x0304);
        lts.add_state(sh);

        let mut done = EncState::new(StateId(3), Phase::Done);
        done.selected_cipher = Some(0x1302);
        done.selected_version = Some(0x0304);
        lts.add_state(done);

        let abort = EncState::new(StateId(4), Phase::Abort);
        lts.add_state(abort);

        // Normal path: Init → CH → SH → Done
        lts.add_transition(EncTransition {
            source: StateId(0),
            target: StateId(1),
            label: EncTransitionLabel::ClientAction {
                action_id: 0,
                ciphers: vec![0x002F, 0x009C, 0x1302],
                version: 0x0304,
            },
            guard: None,
        });
        lts.add_transition(EncTransition {
            source: StateId(1),
            target: StateId(2),
            label: EncTransitionLabel::ServerAction {
                action_id: 0,
                cipher: 0x1302,
                version: 0x0304,
            },
            guard: None,
        });
        lts.add_transition(EncTransition {
            source: StateId(2),
            target: StateId(3),
            label: EncTransitionLabel::Tau,
            guard: None,
        });

        // Adversary paths
        lts.add_transition(EncTransition {
            source: StateId(1),
            target: StateId(4),
            label: EncTransitionLabel::Adversary(AdvAction::Drop),
            guard: None,
        });
        lts.add_transition(EncTransition {
            source: StateId(1),
            target: StateId(2),
            label: EncTransitionLabel::Adversary(AdvAction::Modify { field_id: 0 }),
            guard: None,
        });

        lts
    }

    fn make_downgrade_lts() -> EncNegotiationLTS {
        let init = EncState::new(StateId(0), Phase::Init);
        let mut lts = EncNegotiationLTS::new(init);

        let mut done_strong = EncState::new(StateId(1), Phase::Done);
        done_strong.selected_cipher = Some(0x1302);
        done_strong.selected_version = Some(0x0304);
        lts.add_state(done_strong);

        let mut done_weak = EncState::new(StateId(2), Phase::Done);
        done_weak.selected_cipher = Some(0x002F);
        done_weak.selected_version = Some(0x0303);
        lts.add_state(done_weak);

        // Honest path to strong
        lts.add_transition(EncTransition {
            source: StateId(0),
            target: StateId(1),
            label: EncTransitionLabel::Tau,
            guard: None,
        });

        // Adversary-enabled path to weak
        lts.add_transition(EncTransition {
            source: StateId(0),
            target: StateId(2),
            label: EncTransitionLabel::Adversary(AdvAction::Modify { field_id: 0 }),
            guard: None,
        });

        lts
    }

    #[test]
    fn test_encode_simple_lts() {
        let lts = make_simple_lts();
        let budget = AdversaryBudget::new(3);
        let config = EncoderConfig::default().with_depth(5).with_budget(3);
        let mut encoder = DYEncoder::new(config);
        let result = encoder.encode_lts(&lts, &budget);

        assert!(result.constraint_count() > 0);
        assert!(result.declaration_count() > 0);
        assert!(!result.smtlib2.is_empty());
        assert!(result.smtlib2.contains("(set-logic"));
        assert!(result.smtlib2.contains("(check-sat)"));
    }

    #[test]
    fn test_encode_produces_valid_smtlib() {
        let lts = make_simple_lts();
        let result = encode_lts(&lts, 3, 2);

        assert!(result.smtlib2.contains("(declare-const"));
        assert!(result.smtlib2.contains("(assert"));
        assert!(result.smtlib2.contains("(check-sat)"));
        assert!(result.smtlib2.contains("(exit)"));
    }

    #[test]
    fn test_encode_stats() {
        let lts = make_simple_lts();
        let result = encode_lts(&lts, 5, 3);

        assert_eq!(result.stats.unrolling_depth, 5);
        assert_eq!(result.stats.adversary_budget, 3);
        assert!(result.stats.total_constraints > 0);
        assert!(result.stats.total_variables > 0);
    }

    #[test]
    fn test_encode_with_all_properties() {
        let lts = make_simple_lts();
        let result = encode_lts_all_properties(&lts, 3, 2);
        assert!(result.stats.property_constraints > 0);
    }

    #[test]
    fn test_encode_default() {
        let lts = make_simple_lts();
        let result = encode_lts_default(&lts);
        assert!(result.constraint_count() > 0);
    }

    #[test]
    fn test_encode_downgrade_lts() {
        let lts = make_downgrade_lts();
        let result = encode_lts(&lts, 3, 2);

        // Should have property negation constraints
        let prop_count = result
            .formula
            .constraints
            .iter()
            .filter(|c| matches!(c.origin, ConstraintOrigin::PropertyNegation))
            .count();
        assert!(prop_count > 0);
    }

    #[test]
    fn test_encode_budget_constraints() {
        let lts = make_simple_lts();
        let result = encode_lts(&lts, 5, 2);

        let budget_count = result
            .formula
            .constraints
            .iter()
            .filter(|c| matches!(c.origin, ConstraintOrigin::BudgetBound))
            .count();
        assert!(budget_count > 0);
    }

    #[test]
    fn test_encode_has_transitions() {
        let lts = make_simple_lts();
        let result = encode_lts(&lts, 3, 2);

        let transition_count = result
            .formula
            .constraints
            .iter()
            .filter(|c| matches!(c.origin, ConstraintOrigin::Transition { .. }))
            .count();
        assert!(transition_count > 0);
    }

    #[test]
    fn test_encode_has_initial_state() {
        let lts = make_simple_lts();
        let result = encode_lts(&lts, 3, 2);

        let init_count = result
            .formula
            .constraints
            .iter()
            .filter(|c| matches!(c.origin, ConstraintOrigin::InitialState))
            .count();
        assert!(init_count > 0);
    }

    #[test]
    fn test_encode_depth_bound() {
        let lts = make_simple_lts();
        let result = encode_lts(&lts, 3, 2);

        let depth_count = result
            .formula
            .constraints
            .iter()
            .filter(|c| matches!(c.origin, ConstraintOrigin::DepthBound))
            .count();
        assert!(depth_count > 0);
    }

    #[test]
    fn test_encode_knowledge_constraints() {
        let lts = make_simple_lts();
        let config = EncoderConfig::default()
            .with_depth(3)
            .with_budget(2);
        let budget = config.adversary_budget.clone();
        let mut encoder = DYEncoder::new(config);
        let result = encoder.encode_lts(&lts, &budget);

        let knowledge_count = result
            .formula
            .constraints
            .iter()
            .filter(|c| matches!(c.origin, ConstraintOrigin::KnowledgeAccumulation { .. }))
            .count();
        assert!(knowledge_count > 0);
    }

    #[test]
    fn test_trivially_unsat() {
        let mut formula = SmtFormula::new(1, 1);
        formula.add_constraint(SmtConstraint::new(
            SmtExpr::BoolLit(false),
            ConstraintOrigin::PropertyNegation,
            "false",
        ));
        let result = DyEncodeResult {
            formula,
            stats: EncodingStats::default(),
            optimization_stats: OptimizationStats::default(),
            smtlib2: String::new(),
        };
        assert!(result.is_trivially_unsat());
    }

    #[test]
    fn test_encoder_config_builder() {
        let config = EncoderConfig::default()
            .with_depth(10)
            .with_budget(3)
            .with_properties(vec![PropertyKind::All]);
        assert_eq!(config.max_depth, 10);
        assert_eq!(config.adversary_budget.max_actions, 3);
        assert_eq!(config.properties, vec![PropertyKind::All]);
    }

    #[test]
    fn test_encode_minimal_lts() {
        // Minimal LTS: just init state, no transitions
        let init = EncState::new(StateId(0), Phase::Init);
        let lts = EncNegotiationLTS::new(init);
        let result = encode_lts(&lts, 2, 1);
        assert!(result.constraint_count() > 0);
    }

    #[test]
    fn test_encode_symmetry_breaking() {
        let lts = make_simple_lts();
        let result = encode_lts(&lts, 5, 3);

        let symmetry_count = result
            .formula
            .constraints
            .iter()
            .filter(|c| matches!(c.origin, ConstraintOrigin::SymmetryBreaking))
            .count();
        assert!(symmetry_count > 0);
    }

    #[test]
    fn test_optimization_applied() {
        let lts = make_simple_lts();
        let result = encode_lts(&lts, 3, 2);

        // After optimization, trivial true constraints should be removed
        let trivial_true = result
            .formula
            .constraints
            .iter()
            .filter(|c| matches!(c.formula, SmtExpr::BoolLit(true)))
            .count();
        assert_eq!(trivial_true, 0);
    }

    #[test]
    fn test_smtlib2_parseable() {
        let lts = make_simple_lts();
        let result = encode_lts(&lts, 3, 2);

        // Check basic SMT-LIB2 structure
        let lines: Vec<&str> = result.smtlib2.lines().collect();
        assert!(!lines.is_empty());

        // Should start with comments or set-logic
        let first_non_comment = lines.iter().find(|l| !l.starts_with(';') && !l.is_empty());
        assert!(first_non_comment.is_some());

        // Should end with exit
        let last_line = lines.iter().rev().find(|l| !l.is_empty());
        assert_eq!(last_line, Some(&&"(exit)"));
    }

    #[test]
    fn test_extract_terms_from_lts() {
        let lts = make_simple_lts();
        let config = EncoderConfig::default().with_depth(3);
        let encoder = DYEncoder::new(config);
        let terms = encoder.extract_terms_from_lts(&lts);

        // Should have cipher IDs, version IDs, and nonces
        let cipher_terms = terms
            .iter()
            .filter(|t| matches!(t, DYTerm::CipherSuiteId(_)))
            .count();
        assert!(cipher_terms >= 3); // 0x002F, 0x009C, 0x1302

        let version_terms = terms
            .iter()
            .filter(|t| matches!(t, DYTerm::VersionId(_)))
            .count();
        assert!(version_terms >= 1); // 0x0304
    }

    #[test]
    fn test_incremental_encoding() {
        let lts = make_simple_lts();
        let config = EncoderConfig::default().with_depth(3).with_budget(2);
        let budget = config.adversary_budget.clone();
        let mut encoder = DYEncoder::new(config);

        let result = encoder.encode_lts(&lts, &budget);
        let new_formula = encoder.encode_incremental(&lts, &result.formula, 5);

        assert_eq!(new_formula.depth_bound, 5);
        assert!(new_formula.constraint_count() >= result.constraint_count());
    }
}
