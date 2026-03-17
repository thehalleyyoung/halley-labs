//! Protocol execution unrolling (bounded model checking).
//!
//! Unrolls the NegotiationLTS transitions up to depth k, producing
//! per-time-step state variables, transition guards, and frame conditions.

use crate::{
    AdvAction, ConstraintOrigin, EncNegotiationLTS, EncTransition, EncTransitionLabel, Phase,
    SmtConstraint, SmtDeclaration, SmtExpr, SmtSort, StateId,
};
use crate::bitvector::BvSort;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

// ─── Time step representation ───────────────────────────────────────────

/// A single time step in the unrolled execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeStep {
    pub step: u32,
    /// State variable names for this step.
    pub state_var: String,
    pub cipher_var: String,
    pub version_var: String,
    pub phase_var: String,
    /// Active adversary action at this step.
    pub adv_action_var: String,
    /// Whether the adversary acts at this step.
    pub adv_active_var: String,
    /// Message variable for this step.
    pub message_var: String,
    /// Extension set variable.
    pub extension_set_var: String,
    /// Offered ciphers set variable.
    pub offered_ciphers_var: String,
    /// Whether this step is in a terminal state.
    pub terminal_var: String,
}

impl TimeStep {
    pub fn new(step: u32) -> Self {
        TimeStep {
            step,
            state_var: format!("state_t{}", step),
            cipher_var: format!("cipher_t{}", step),
            version_var: format!("version_t{}", step),
            phase_var: format!("phase_t{}", step),
            adv_action_var: format!("adv_action_t{}", step),
            adv_active_var: format!("adv_active_t{}", step),
            message_var: format!("msg_t{}", step),
            extension_set_var: format!("ext_set_t{}", step),
            offered_ciphers_var: format!("offered_t{}", step),
            terminal_var: format!("terminal_t{}", step),
        }
    }

    /// All SMT declarations for this time step.
    pub fn declarations(&self, state_bv_width: u32) -> Vec<SmtDeclaration> {
        vec![
            SmtDeclaration::DeclareConst {
                name: self.state_var.clone(),
                sort: SmtSort::BitVec(state_bv_width),
            },
            SmtDeclaration::DeclareConst {
                name: self.cipher_var.clone(),
                sort: BvSort::cipher_suite().to_smt_sort(),
            },
            SmtDeclaration::DeclareConst {
                name: self.version_var.clone(),
                sort: BvSort::version().to_smt_sort(),
            },
            SmtDeclaration::DeclareConst {
                name: self.phase_var.clone(),
                sort: BvSort::phase().to_smt_sort(),
            },
            SmtDeclaration::DeclareConst {
                name: self.adv_action_var.clone(),
                sort: BvSort::action_type().to_smt_sort(),
            },
            SmtDeclaration::DeclareConst {
                name: self.adv_active_var.clone(),
                sort: SmtSort::Bool,
            },
            SmtDeclaration::DeclareConst {
                name: self.message_var.clone(),
                sort: SmtSort::BitVec(256),
            },
            SmtDeclaration::DeclareConst {
                name: self.extension_set_var.clone(),
                sort: SmtSort::Array(
                    Box::new(SmtSort::BitVec(16)),
                    Box::new(SmtSort::Bool),
                ),
            },
            SmtDeclaration::DeclareConst {
                name: self.offered_ciphers_var.clone(),
                sort: SmtSort::Array(
                    Box::new(SmtSort::BitVec(16)),
                    Box::new(SmtSort::Bool),
                ),
            },
            SmtDeclaration::DeclareConst {
                name: self.terminal_var.clone(),
                sort: SmtSort::Bool,
            },
        ]
    }

    /// State variable expression.
    pub fn state_expr(&self) -> SmtExpr {
        SmtExpr::var(&self.state_var)
    }

    pub fn cipher_expr(&self) -> SmtExpr {
        SmtExpr::var(&self.cipher_var)
    }

    pub fn version_expr(&self) -> SmtExpr {
        SmtExpr::var(&self.version_var)
    }

    pub fn phase_expr(&self) -> SmtExpr {
        SmtExpr::var(&self.phase_var)
    }

    pub fn adv_active_expr(&self) -> SmtExpr {
        SmtExpr::var(&self.adv_active_var)
    }

    pub fn adv_action_expr(&self) -> SmtExpr {
        SmtExpr::var(&self.adv_action_var)
    }

    pub fn terminal_expr(&self) -> SmtExpr {
        SmtExpr::var(&self.terminal_var)
    }
}

// ─── Unrolling Engine ───────────────────────────────────────────────────

/// Configuration for the unrolling engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnrollingConfig {
    pub max_depth: u32,
    pub state_bv_width: u32,
    pub enable_frame_conditions: bool,
    pub enable_terminal_detection: bool,
    pub enable_phase_monotonicity: bool,
}

impl Default for UnrollingConfig {
    fn default() -> Self {
        UnrollingConfig {
            max_depth: 20,
            state_bv_width: 8,
            enable_frame_conditions: true,
            enable_terminal_detection: true,
            enable_phase_monotonicity: true,
        }
    }
}

/// Engine for unrolling LTS transitions up to bounded depth.
#[derive(Debug, Clone)]
pub struct UnrollingEngine {
    config: UnrollingConfig,
    time_steps: Vec<TimeStep>,
    state_encoding: std::collections::BTreeMap<StateId, u64>,
}

impl UnrollingEngine {
    pub fn new(config: UnrollingConfig) -> Self {
        UnrollingEngine {
            config,
            time_steps: Vec::new(),
            state_encoding: std::collections::BTreeMap::new(),
        }
    }

    pub fn with_depth(depth: u32) -> Self {
        Self::new(UnrollingConfig {
            max_depth: depth,
            ..Default::default()
        })
    }

    /// Initialize time steps and compute state encoding from the LTS.
    pub fn initialize(&mut self, lts: &EncNegotiationLTS) {
        // Assign bitvector values to states
        self.state_encoding.clear();
        for (i, &state_id) in lts.states.keys().enumerate() {
            self.state_encoding.insert(state_id, i as u64);
        }

        // Compute needed bit width for state encoding
        let num_states = lts.state_count();
        let needed_bits = if num_states <= 1 {
            1
        } else {
            64 - ((num_states as u64) - 1).leading_zeros()
        };
        self.config.state_bv_width = needed_bits;

        // Create time steps
        self.time_steps.clear();
        for step in 0..=self.config.max_depth {
            self.time_steps.push(TimeStep::new(step));
        }
    }

    /// Generate all declarations for the unrolled encoding.
    pub fn declarations(&self) -> Vec<SmtDeclaration> {
        let mut decls = Vec::new();
        for ts in &self.time_steps {
            decls.extend(ts.declarations(self.config.state_bv_width));
        }
        decls
    }

    /// Encode the initial state constraint.
    pub fn encode_initial_state(&self, lts: &EncNegotiationLTS) -> Vec<SmtConstraint> {
        let mut constraints = Vec::new();
        let ts = &self.time_steps[0];
        let init_state = lts.initial_state;

        // State variable equals initial state encoding
        if let Some(&encoding) = self.state_encoding.get(&init_state) {
            constraints.push(SmtConstraint::new(
                SmtExpr::eq(
                    ts.state_expr(),
                    SmtExpr::bv_lit(encoding, self.config.state_bv_width),
                ),
                ConstraintOrigin::InitialState,
                "initial_state",
            ));
        }

        // Initial phase is Init
        constraints.push(SmtConstraint::new(
            SmtExpr::eq(ts.phase_expr(), SmtExpr::bv_lit(0, 4)),
            ConstraintOrigin::InitialState,
            "initial_phase",
        ));

        // Not terminal initially
        constraints.push(SmtConstraint::new(
            SmtExpr::not(ts.terminal_expr()),
            ConstraintOrigin::InitialState,
            "initial_not_terminal",
        ));

        // Initial offered ciphers from the initial state
        if let Some(init) = lts.states.get(&init_state) {
            if let Some(cipher) = init.selected_cipher {
                constraints.push(SmtConstraint::new(
                    SmtExpr::eq(ts.cipher_expr(), SmtExpr::bv_lit(cipher as u64, 16)),
                    ConstraintOrigin::InitialState,
                    "initial_cipher",
                ));
            }
            if let Some(version) = init.selected_version {
                constraints.push(SmtConstraint::new(
                    SmtExpr::eq(ts.version_expr(), SmtExpr::bv_lit(version as u64, 16)),
                    ConstraintOrigin::InitialState,
                    "initial_version",
                ));
            }
        }

        // No adversary action at step 0
        constraints.push(SmtConstraint::new(
            SmtExpr::not(ts.adv_active_expr()),
            ConstraintOrigin::InitialState,
            "no_initial_adv_action",
        ));

        constraints
    }

    /// Encode transition constraints for a single time step.
    pub fn encode_transitions_at_step(
        &self,
        lts: &EncNegotiationLTS,
        step: u32,
    ) -> Vec<SmtConstraint> {
        if step as usize >= self.time_steps.len() - 1 {
            return Vec::new();
        }

        let ts_curr = &self.time_steps[step as usize];
        let ts_next = &self.time_steps[(step + 1) as usize];
        let mut constraints = Vec::new();

        // For each transition in the LTS, encode it as an implication:
        // (state_curr == src) ∧ guard → (state_next == tgt) ∧ effects
        let mut transition_disjuncts = Vec::new();

        for trans in &lts.transitions {
            let src_enc = match self.state_encoding.get(&trans.source) {
                Some(&e) => e,
                None => continue,
            };
            let tgt_enc = match self.state_encoding.get(&trans.target) {
                Some(&e) => e,
                None => continue,
            };

            let src_match = SmtExpr::eq(
                ts_curr.state_expr(),
                SmtExpr::bv_lit(src_enc, self.config.state_bv_width),
            );

            let tgt_assign = SmtExpr::eq(
                ts_next.state_expr(),
                SmtExpr::bv_lit(tgt_enc, self.config.state_bv_width),
            );

            let mut conjuncts = vec![src_match.clone()];

            // Add guard if present
            if let Some(ref guard) = trans.guard {
                conjuncts.push(guard.clone());
            }

            // Build effect of this transition
            let mut effects = vec![tgt_assign];
            effects.extend(self.encode_transition_effects(trans, ts_curr, ts_next, lts));

            // Is adversary transition?
            let is_adv = trans.label.is_adversary();
            if is_adv {
                conjuncts.push(ts_curr.adv_active_expr());
                let action_idx = match &trans.label {
                    EncTransitionLabel::Adversary(a) => a.action_index(),
                    _ => 5,
                };
                conjuncts.push(SmtExpr::eq(
                    ts_curr.adv_action_expr(),
                    SmtExpr::bv_lit(action_idx as u64, 4),
                ));
            } else {
                conjuncts.push(SmtExpr::not(ts_curr.adv_active_expr()));
            }

            let guard_conj = SmtExpr::and(conjuncts);
            let effect_conj = SmtExpr::and(effects);
            transition_disjuncts.push(SmtExpr::and(vec![guard_conj, effect_conj]));
        }

        // If already terminal, stay terminal (stutter)
        let terminal_stutter = self.encode_terminal_stutter(ts_curr, ts_next);
        transition_disjuncts.push(terminal_stutter);

        // At each step, exactly one transition fires
        if !transition_disjuncts.is_empty() {
            constraints.push(SmtConstraint::new(
                SmtExpr::or(transition_disjuncts),
                ConstraintOrigin::Transition {
                    from: 0,
                    to: 0,
                    step,
                },
                format!("transition_step_{}", step),
            ));
        }

        constraints
    }

    /// Encode the effects of taking a particular transition.
    fn encode_transition_effects(
        &self,
        trans: &EncTransition,
        _ts_curr: &TimeStep,
        ts_next: &TimeStep,
        lts: &EncNegotiationLTS,
    ) -> Vec<SmtExpr> {
        let mut effects = Vec::new();

        // Update phase based on target state
        if let Some(tgt_state) = lts.states.get(&trans.target) {
            effects.push(SmtExpr::eq(
                ts_next.phase_expr(),
                SmtExpr::bv_lit(tgt_state.phase.to_index() as u64, 4),
            ));

            // Update terminal flag
            if tgt_state.is_terminal {
                effects.push(ts_next.terminal_expr());
            } else {
                effects.push(SmtExpr::not(ts_next.terminal_expr()));
            }

            // Update selected cipher if the target has one
            if let Some(cipher) = tgt_state.selected_cipher {
                effects.push(SmtExpr::eq(
                    ts_next.cipher_expr(),
                    SmtExpr::bv_lit(cipher as u64, 16),
                ));
            }

            // Update selected version
            if let Some(version) = tgt_state.selected_version {
                effects.push(SmtExpr::eq(
                    ts_next.version_expr(),
                    SmtExpr::bv_lit(version as u64, 16),
                ));
            }
        }

        // Label-specific effects
        match &trans.label {
            EncTransitionLabel::ServerAction { cipher, version, .. } => {
                effects.push(SmtExpr::eq(
                    ts_next.cipher_expr(),
                    SmtExpr::bv_lit(*cipher as u64, 16),
                ));
                effects.push(SmtExpr::eq(
                    ts_next.version_expr(),
                    SmtExpr::bv_lit(*version as u64, 16),
                ));
            }
            EncTransitionLabel::Adversary(AdvAction::Modify { .. }) => {
                // Modified cipher/version are unconstrained (adversary chooses)
            }
            _ => {}
        }

        effects
    }

    /// Encode terminal state stutter: if terminal, state stays the same.
    fn encode_terminal_stutter(&self, ts_curr: &TimeStep, ts_next: &TimeStep) -> SmtExpr {
        SmtExpr::and(vec![
            ts_curr.terminal_expr(),
            SmtExpr::eq(ts_next.state_expr(), ts_curr.state_expr()),
            SmtExpr::eq(ts_next.cipher_expr(), ts_curr.cipher_expr()),
            SmtExpr::eq(ts_next.version_expr(), ts_curr.version_expr()),
            SmtExpr::eq(ts_next.phase_expr(), ts_curr.phase_expr()),
            ts_next.terminal_expr(),
            SmtExpr::not(ts_next.adv_active_expr()),
        ])
    }

    /// Encode frame conditions: variables not modified by a transition retain their value.
    pub fn encode_frame_conditions(&self, step: u32) -> Vec<SmtConstraint> {
        if !self.config.enable_frame_conditions {
            return Vec::new();
        }
        // Frame conditions are implicitly handled by the transition encoding
        // since we require exactly one disjunct. Additional frame conditions
        // for extension sets:
        let ts_curr = &self.time_steps[step as usize];
        let ts_next = &self.time_steps[(step + 1) as usize];

        vec![SmtConstraint::new(
            SmtExpr::implies(
                SmtExpr::not(ts_curr.adv_active_expr()),
                SmtExpr::eq(
                    SmtExpr::var(&ts_next.extension_set_var),
                    SmtExpr::var(&ts_curr.extension_set_var),
                ),
            ),
            ConstraintOrigin::FrameCondition { step },
            format!("frame_extensions_step_{}", step),
        )]
    }

    /// Encode phase monotonicity constraints.
    pub fn encode_phase_monotonicity(&self) -> Vec<SmtConstraint> {
        if !self.config.enable_phase_monotonicity {
            return Vec::new();
        }

        let mut constraints = Vec::new();
        for step in 0..self.config.max_depth {
            let curr = &self.time_steps[step as usize];
            let next = &self.time_steps[(step + 1) as usize];
            constraints.push(SmtConstraint::new(
                SmtExpr::implies(
                    SmtExpr::not(curr.terminal_expr()),
                    SmtExpr::bv_ule(curr.phase_expr(), next.phase_expr()),
                ),
                ConstraintOrigin::Transition { from: 0, to: 0, step },
                format!("phase_monotonicity_step_{}", step),
            ));
        }
        constraints
    }

    /// Encode terminal state detection.
    pub fn encode_terminal_detection(
        &self,
        lts: &EncNegotiationLTS,
    ) -> Vec<SmtConstraint> {
        if !self.config.enable_terminal_detection {
            return Vec::new();
        }

        let terminal_states = lts.terminal_states();
        let mut constraints = Vec::new();

        for ts in &self.time_steps {
            let is_terminal_disjuncts: Vec<SmtExpr> = terminal_states
                .iter()
                .filter_map(|sid| {
                    self.state_encoding.get(sid).map(|&enc| {
                        SmtExpr::eq(
                            ts.state_expr(),
                            SmtExpr::bv_lit(enc, self.config.state_bv_width),
                        )
                    })
                })
                .collect();

            if !is_terminal_disjuncts.is_empty() {
                constraints.push(SmtConstraint::new(
                    SmtExpr::eq(ts.terminal_expr(), SmtExpr::or(is_terminal_disjuncts)),
                    ConstraintOrigin::Transition { from: 0, to: 0, step: ts.step },
                    format!("terminal_detection_step_{}", ts.step),
                ));
            }
        }
        constraints
    }

    /// Encode all step transitions for the complete unrolling.
    pub fn encode_all_transitions(
        &self,
        lts: &EncNegotiationLTS,
    ) -> Vec<SmtConstraint> {
        let mut constraints = Vec::new();
        for step in 0..self.config.max_depth {
            constraints.extend(self.encode_transitions_at_step(lts, step));
            constraints.extend(self.encode_frame_conditions(step));
        }
        constraints.extend(self.encode_phase_monotonicity());
        constraints.extend(self.encode_terminal_detection(lts));
        constraints
    }

    /// Encode that execution must reach a terminal state by depth k.
    pub fn encode_termination_requirement(&self) -> SmtConstraint {
        let last = &self.time_steps[self.config.max_depth as usize];
        SmtConstraint::new(
            last.terminal_expr(),
            ConstraintOrigin::DepthBound,
            "must_terminate",
        )
    }

    /// Get the time step for a given index.
    pub fn time_step(&self, step: u32) -> Option<&TimeStep> {
        self.time_steps.get(step as usize)
    }

    /// Get all time steps.
    pub fn time_steps(&self) -> &[TimeStep] {
        &self.time_steps
    }

    pub fn depth(&self) -> u32 {
        self.config.max_depth
    }

    pub fn state_bv_width(&self) -> u32 {
        self.config.state_bv_width
    }

    pub fn state_encoding(&self) -> &std::collections::BTreeMap<StateId, u64> {
        &self.state_encoding
    }

    pub fn config(&self) -> &UnrollingConfig {
        &self.config
    }

    /// Encode valid state constraint: state var must be one of the known states.
    pub fn encode_valid_states(&self, lts: &EncNegotiationLTS) -> Vec<SmtConstraint> {
        let mut constraints = Vec::new();
        let valid_values: Vec<SmtExpr> = self
            .state_encoding
            .values()
            .map(|&enc| SmtExpr::bv_lit(enc, self.config.state_bv_width))
            .collect();

        for ts in &self.time_steps {
            let valid = SmtExpr::or(
                valid_values
                    .iter()
                    .map(|v| SmtExpr::eq(ts.state_expr(), v.clone()))
                    .collect(),
            );
            constraints.push(SmtConstraint::new(
                valid,
                ConstraintOrigin::Transition { from: 0, to: 0, step: ts.step },
                format!("valid_state_step_{}", ts.step),
            ));
        }
        constraints
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EncState, Phase};

    fn make_test_lts() -> EncNegotiationLTS {
        let init = EncState::new(StateId(0), Phase::Init);
        let mut lts = EncNegotiationLTS::new(init);

        let mut ch_sent = EncState::new(StateId(1), Phase::ClientHelloSent);
        ch_sent.offered_ciphers.insert(0x002F);
        ch_sent.offered_ciphers.insert(0x009C);
        lts.add_state(ch_sent);

        let mut sh_recv = EncState::new(StateId(2), Phase::ServerHelloReceived);
        sh_recv.selected_cipher = Some(0x009C);
        sh_recv.selected_version = Some(0x0303);
        lts.add_state(sh_recv);

        let mut done = EncState::new(StateId(3), Phase::Done);
        done.selected_cipher = Some(0x009C);
        done.selected_version = Some(0x0303);
        lts.add_state(done);

        let abort = EncState::new(StateId(4), Phase::Abort);
        lts.add_state(abort);

        lts.add_transition(EncTransition {
            source: StateId(0),
            target: StateId(1),
            label: EncTransitionLabel::ClientAction {
                action_id: 0,
                ciphers: vec![0x002F, 0x009C],
                version: 0x0303,
            },
            guard: None,
        });

        lts.add_transition(EncTransition {
            source: StateId(1),
            target: StateId(2),
            label: EncTransitionLabel::ServerAction {
                action_id: 0,
                cipher: 0x009C,
                version: 0x0303,
            },
            guard: None,
        });

        lts.add_transition(EncTransition {
            source: StateId(2),
            target: StateId(3),
            label: EncTransitionLabel::Tau,
            guard: None,
        });

        // Adversary drop transition
        lts.add_transition(EncTransition {
            source: StateId(1),
            target: StateId(4),
            label: EncTransitionLabel::Adversary(AdvAction::Drop),
            guard: None,
        });

        lts
    }

    #[test]
    fn test_time_step_creation() {
        let ts = TimeStep::new(3);
        assert_eq!(ts.step, 3);
        assert_eq!(ts.state_var, "state_t3");
        assert_eq!(ts.cipher_var, "cipher_t3");
    }

    #[test]
    fn test_time_step_declarations() {
        let ts = TimeStep::new(0);
        let decls = ts.declarations(8);
        assert!(decls.len() >= 8);
    }

    #[test]
    fn test_unrolling_engine_init() {
        let lts = make_test_lts();
        let mut engine = UnrollingEngine::with_depth(5);
        engine.initialize(&lts);

        assert_eq!(engine.time_steps().len(), 6); // 0..=5
        assert_eq!(engine.state_encoding().len(), 5);
    }

    #[test]
    fn test_initial_state_constraints() {
        let lts = make_test_lts();
        let mut engine = UnrollingEngine::with_depth(5);
        engine.initialize(&lts);

        let constraints = engine.encode_initial_state(&lts);
        assert!(!constraints.is_empty());

        let labels: Vec<&str> = constraints.iter().map(|c| c.label.as_str()).collect();
        assert!(labels.contains(&"initial_state"));
        assert!(labels.contains(&"initial_phase"));
    }

    #[test]
    fn test_transition_encoding() {
        let lts = make_test_lts();
        let mut engine = UnrollingEngine::with_depth(5);
        engine.initialize(&lts);

        let constraints = engine.encode_transitions_at_step(&lts, 0);
        assert!(!constraints.is_empty());
    }

    #[test]
    fn test_all_transitions() {
        let lts = make_test_lts();
        let mut engine = UnrollingEngine::with_depth(3);
        engine.initialize(&lts);

        let constraints = engine.encode_all_transitions(&lts);
        assert!(!constraints.is_empty());
    }

    #[test]
    fn test_termination_requirement() {
        let lts = make_test_lts();
        let mut engine = UnrollingEngine::with_depth(5);
        engine.initialize(&lts);

        let constraint = engine.encode_termination_requirement();
        let s = format!("{}", constraint.formula);
        assert!(s.contains("terminal_t5"));
    }

    #[test]
    fn test_phase_monotonicity() {
        let lts = make_test_lts();
        let mut engine = UnrollingEngine::with_depth(3);
        engine.initialize(&lts);

        let constraints = engine.encode_phase_monotonicity();
        assert_eq!(constraints.len(), 3);
    }

    #[test]
    fn test_terminal_detection() {
        let lts = make_test_lts();
        let mut engine = UnrollingEngine::with_depth(3);
        engine.initialize(&lts);

        let constraints = engine.encode_terminal_detection(&lts);
        assert!(!constraints.is_empty());
    }

    #[test]
    fn test_valid_states() {
        let lts = make_test_lts();
        let mut engine = UnrollingEngine::with_depth(2);
        engine.initialize(&lts);

        let constraints = engine.encode_valid_states(&lts);
        assert_eq!(constraints.len(), 3); // steps 0, 1, 2
    }

    #[test]
    fn test_frame_conditions() {
        let lts = make_test_lts();
        let mut engine = UnrollingEngine::with_depth(3);
        engine.initialize(&lts);

        let constraints = engine.encode_frame_conditions(0);
        assert!(!constraints.is_empty());
    }

    #[test]
    fn test_declarations() {
        let lts = make_test_lts();
        let mut engine = UnrollingEngine::with_depth(2);
        engine.initialize(&lts);

        let decls = engine.declarations();
        assert!(decls.len() >= 20);
    }

    #[test]
    fn test_state_encoding_roundtrip() {
        let lts = make_test_lts();
        let mut engine = UnrollingEngine::with_depth(2);
        engine.initialize(&lts);

        let encoding = engine.state_encoding();
        for (&state_id, &_enc) in encoding {
            assert!(lts.states.contains_key(&state_id));
        }
    }
}
