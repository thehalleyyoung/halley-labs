//! Adversary-specific SMT encoding.
//!
//! Encodes the bounded Dolev-Yao adversary model:
//! - Boolean action variables per time step
//! - Budget tracking via integer sum constraints
//! - Message injection, modification, drop, and interception

use crate::bitvector::{BvSort, CardinalityEncoder};
use crate::dolev_yao::{DYTerm, KnowledgeEncoder};
use crate::unrolling::{TimeStep, UnrollingEngine};
use crate::{
    AdvAction, AdversaryBudget, ConstraintOrigin, SmtConstraint, SmtDeclaration, SmtExpr,
    SmtSort,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

// ─── Adversary encoder configuration ────────────────────────────────────

/// Configuration for the adversary encoding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdversaryEncoderConfig {
    pub enable_injection: bool,
    pub enable_modification: bool,
    pub enable_drop: bool,
    pub enable_interception: bool,
    pub enable_replay: bool,
    pub enable_symmetry_breaking: bool,
    pub max_derivation_depth: u32,
}

impl Default for AdversaryEncoderConfig {
    fn default() -> Self {
        AdversaryEncoderConfig {
            enable_injection: true,
            enable_modification: true,
            enable_drop: true,
            enable_interception: true,
            enable_replay: false,
            enable_symmetry_breaking: true,
            max_derivation_depth: 3,
        }
    }
}

// ─── Adversary action variables ─────────────────────────────────────────

/// Variables tracking adversary actions at a single time step.
#[derive(Debug, Clone)]
pub struct StepActionVars {
    pub step: u32,
    pub active: String,
    pub action_type: String,
    pub drop_flag: String,
    pub intercept_flag: String,
    pub inject_flag: String,
    pub modify_flag: String,
    pub replay_flag: String,
    pub inject_payload: String,
    pub modify_field: String,
    pub modify_value: String,
}

impl StepActionVars {
    pub fn new(step: u32) -> Self {
        StepActionVars {
            step,
            active: format!("adv_active_t{}", step),
            action_type: format!("adv_action_t{}", step),
            drop_flag: format!("adv_drop_t{}", step),
            intercept_flag: format!("adv_intercept_t{}", step),
            inject_flag: format!("adv_inject_t{}", step),
            modify_flag: format!("adv_modify_t{}", step),
            replay_flag: format!("adv_replay_t{}", step),
            inject_payload: format!("adv_inject_payload_t{}", step),
            modify_field: format!("adv_modify_field_t{}", step),
            modify_value: format!("adv_modify_value_t{}", step),
        }
    }

    /// SMT declarations for this step's action variables.
    pub fn declarations(&self) -> Vec<SmtDeclaration> {
        vec![
            SmtDeclaration::DeclareConst {
                name: self.active.clone(),
                sort: SmtSort::Bool,
            },
            SmtDeclaration::DeclareConst {
                name: self.action_type.clone(),
                sort: BvSort::action_type().to_smt_sort(),
            },
            SmtDeclaration::DeclareConst {
                name: self.drop_flag.clone(),
                sort: SmtSort::Bool,
            },
            SmtDeclaration::DeclareConst {
                name: self.intercept_flag.clone(),
                sort: SmtSort::Bool,
            },
            SmtDeclaration::DeclareConst {
                name: self.inject_flag.clone(),
                sort: SmtSort::Bool,
            },
            SmtDeclaration::DeclareConst {
                name: self.modify_flag.clone(),
                sort: SmtSort::Bool,
            },
            SmtDeclaration::DeclareConst {
                name: self.replay_flag.clone(),
                sort: SmtSort::Bool,
            },
            SmtDeclaration::DeclareConst {
                name: self.inject_payload.clone(),
                sort: SmtSort::BitVec(256),
            },
            SmtDeclaration::DeclareConst {
                name: self.modify_field.clone(),
                sort: SmtSort::BitVec(8),
            },
            SmtDeclaration::DeclareConst {
                name: self.modify_value.clone(),
                sort: SmtSort::BitVec(16),
            },
        ]
    }

    /// Constraint: exactly one action flag is set when active.
    pub fn exactly_one_action(&self, config: &AdversaryEncoderConfig) -> SmtExpr {
        let mut flags = Vec::new();
        if config.enable_drop {
            flags.push(SmtExpr::var(&self.drop_flag));
        }
        if config.enable_interception {
            flags.push(SmtExpr::var(&self.intercept_flag));
        }
        if config.enable_injection {
            flags.push(SmtExpr::var(&self.inject_flag));
        }
        if config.enable_modification {
            flags.push(SmtExpr::var(&self.modify_flag));
        }
        if config.enable_replay {
            flags.push(SmtExpr::var(&self.replay_flag));
        }

        if flags.is_empty() {
            return SmtExpr::not(SmtExpr::var(&self.active));
        }

        let active = SmtExpr::var(&self.active);
        let exactly_one = SmtExpr::and(vec![
            SmtExpr::or(flags.clone()),
            CardinalityEncoder::at_most_one_pairwise(&flags),
        ]);

        SmtExpr::implies(active, exactly_one)
    }

    /// Constraint: if not active, no flags are set.
    pub fn inactive_means_no_flags(&self) -> SmtExpr {
        let not_active = SmtExpr::not(SmtExpr::var(&self.active));
        SmtExpr::implies(
            not_active,
            SmtExpr::and(vec![
                SmtExpr::not(SmtExpr::var(&self.drop_flag)),
                SmtExpr::not(SmtExpr::var(&self.intercept_flag)),
                SmtExpr::not(SmtExpr::var(&self.inject_flag)),
                SmtExpr::not(SmtExpr::var(&self.modify_flag)),
                SmtExpr::not(SmtExpr::var(&self.replay_flag)),
            ]),
        )
    }

    /// Constraint linking action_type bitvector to boolean flags.
    pub fn action_type_consistency(&self) -> SmtExpr {
        SmtExpr::and(vec![
            SmtExpr::implies(
                SmtExpr::var(&self.drop_flag),
                SmtExpr::eq(
                    SmtExpr::var(&self.action_type),
                    SmtExpr::bv_lit(AdvAction::Drop.action_index() as u64, 4),
                ),
            ),
            SmtExpr::implies(
                SmtExpr::var(&self.intercept_flag),
                SmtExpr::eq(
                    SmtExpr::var(&self.action_type),
                    SmtExpr::bv_lit(AdvAction::Intercept.action_index() as u64, 4),
                ),
            ),
            SmtExpr::implies(
                SmtExpr::var(&self.inject_flag),
                SmtExpr::eq(
                    SmtExpr::var(&self.action_type),
                    SmtExpr::bv_lit(AdvAction::Inject { payload_id: 0 }.action_index() as u64, 4),
                ),
            ),
            SmtExpr::implies(
                SmtExpr::var(&self.modify_flag),
                SmtExpr::eq(
                    SmtExpr::var(&self.action_type),
                    SmtExpr::bv_lit(AdvAction::Modify { field_id: 0 }.action_index() as u64, 4),
                ),
            ),
        ])
    }
}

// ─── Adversary Encoder ──────────────────────────────────────────────────

/// Encodes the bounded Dolev-Yao adversary as SMT constraints.
#[derive(Debug, Clone)]
pub struct AdversaryEncoder {
    config: AdversaryEncoderConfig,
    budget: AdversaryBudget,
    step_vars: Vec<StepActionVars>,
}

impl AdversaryEncoder {
    pub fn new(budget: AdversaryBudget, config: AdversaryEncoderConfig) -> Self {
        AdversaryEncoder {
            config,
            budget,
            step_vars: Vec::new(),
        }
    }

    pub fn with_default_config(budget: AdversaryBudget) -> Self {
        Self::new(budget, AdversaryEncoderConfig::default())
    }

    /// Initialize step action variables for the given depth.
    pub fn initialize(&mut self, depth: u32) {
        self.step_vars.clear();
        for step in 0..=depth {
            self.step_vars.push(StepActionVars::new(step));
        }
    }

    /// Generate all declarations for adversary variables.
    pub fn declarations(&self) -> Vec<SmtDeclaration> {
        let mut decls = Vec::new();
        for sv in &self.step_vars {
            decls.extend(sv.declarations());
        }
        decls
    }

    /// Encode the global budget constraint: sum of active steps ≤ max_actions.
    pub fn encode_budget_constraint(&self) -> SmtConstraint {
        let active_vars: Vec<SmtExpr> = self
            .step_vars
            .iter()
            .map(|sv| SmtExpr::var(&sv.active))
            .collect();

        SmtConstraint::new(
            CardinalityEncoder::at_most_k(&active_vars, self.budget.max_actions),
            ConstraintOrigin::BudgetBound,
            "adversary_budget",
        )
    }

    /// Encode per-action-type budget constraints.
    pub fn encode_per_action_budgets(&self) -> Vec<SmtConstraint> {
        let mut constraints = Vec::new();

        // Drop budget
        if self.config.enable_drop {
            let drop_vars: Vec<SmtExpr> = self
                .step_vars
                .iter()
                .map(|sv| SmtExpr::var(&sv.drop_flag))
                .collect();
            constraints.push(SmtConstraint::new(
                CardinalityEncoder::at_most_k(&drop_vars, self.budget.max_drops),
                ConstraintOrigin::BudgetBound,
                "drop_budget",
            ));
        }

        // Inject budget
        if self.config.enable_injection {
            let inject_vars: Vec<SmtExpr> = self
                .step_vars
                .iter()
                .map(|sv| SmtExpr::var(&sv.inject_flag))
                .collect();
            constraints.push(SmtConstraint::new(
                CardinalityEncoder::at_most_k(&inject_vars, self.budget.max_injects),
                ConstraintOrigin::BudgetBound,
                "inject_budget",
            ));
        }

        // Modify budget
        if self.config.enable_modification {
            let modify_vars: Vec<SmtExpr> = self
                .step_vars
                .iter()
                .map(|sv| SmtExpr::var(&sv.modify_flag))
                .collect();
            constraints.push(SmtConstraint::new(
                CardinalityEncoder::at_most_k(&modify_vars, self.budget.max_modifies),
                ConstraintOrigin::BudgetBound,
                "modify_budget",
            ));
        }

        // Intercept budget
        if self.config.enable_interception {
            let intercept_vars: Vec<SmtExpr> = self
                .step_vars
                .iter()
                .map(|sv| SmtExpr::var(&sv.intercept_flag))
                .collect();
            constraints.push(SmtConstraint::new(
                CardinalityEncoder::at_most_k(&intercept_vars, self.budget.max_intercepts),
                ConstraintOrigin::BudgetBound,
                "intercept_budget",
            ));
        }

        constraints
    }

    /// Encode action consistency constraints for all steps.
    pub fn encode_action_consistency(&self) -> Vec<SmtConstraint> {
        let mut constraints = Vec::new();
        for sv in &self.step_vars {
            constraints.push(SmtConstraint::new(
                sv.exactly_one_action(&self.config),
                ConstraintOrigin::AdversaryAction {
                    action_idx: sv.step,
                    step: sv.step,
                },
                format!("action_exactly_one_t{}", sv.step),
            ));

            constraints.push(SmtConstraint::new(
                sv.inactive_means_no_flags(),
                ConstraintOrigin::AdversaryAction {
                    action_idx: sv.step,
                    step: sv.step,
                },
                format!("inactive_no_flags_t{}", sv.step),
            ));

            constraints.push(SmtConstraint::new(
                sv.action_type_consistency(),
                ConstraintOrigin::AdversaryAction {
                    action_idx: sv.step,
                    step: sv.step,
                },
                format!("action_type_consistency_t{}", sv.step),
            ));
        }
        constraints
    }

    /// Encode injection constraints: injected message must be derivable
    /// from adversary knowledge.
    pub fn encode_injection_constraints(
        &self,
        knowledge_encoder: &KnowledgeEncoder,
        derivable_terms: &[DYTerm],
    ) -> Vec<SmtConstraint> {
        let mut constraints = Vec::new();

        for sv in &self.step_vars {
            if !self.config.enable_injection {
                continue;
            }

            // If injecting, the payload must match some derivable term
            let derivable_disjuncts: Vec<SmtExpr> = derivable_terms
                .iter()
                .map(|term| {
                    let knows = SmtExpr::var(knowledge_encoder.knows_var(term, sv.step));
                    knows
                })
                .collect();

            if !derivable_disjuncts.is_empty() {
                constraints.push(SmtConstraint::new(
                    SmtExpr::implies(
                        SmtExpr::var(&sv.inject_flag),
                        SmtExpr::or(derivable_disjuncts),
                    ),
                    ConstraintOrigin::AdversaryAction {
                        action_idx: sv.step,
                        step: sv.step,
                    },
                    format!("inject_derivable_t{}", sv.step),
                ));
            }
        }
        constraints
    }

    /// Encode modification constraints: modified value must be derivable.
    pub fn encode_modification_constraints(
        &self,
        knowledge_encoder: &KnowledgeEncoder,
        modifiable_terms: &[DYTerm],
    ) -> Vec<SmtConstraint> {
        let mut constraints = Vec::new();

        for sv in &self.step_vars {
            if !self.config.enable_modification {
                continue;
            }

            // Modified value comes from knowledge
            let derivable_disjuncts: Vec<SmtExpr> = modifiable_terms
                .iter()
                .map(|term| SmtExpr::var(knowledge_encoder.knows_var(term, sv.step)))
                .collect();

            if !derivable_disjuncts.is_empty() {
                constraints.push(SmtConstraint::new(
                    SmtExpr::implies(
                        SmtExpr::var(&sv.modify_flag),
                        SmtExpr::or(derivable_disjuncts),
                    ),
                    ConstraintOrigin::AdversaryAction {
                        action_idx: sv.step,
                        step: sv.step,
                    },
                    format!("modify_derivable_t{}", sv.step),
                ));
            }
        }
        constraints
    }

    /// Encode interception: if intercepting, message is added to knowledge.
    pub fn encode_interception_effects(
        &self,
        knowledge_encoder: &KnowledgeEncoder,
        message_terms: &[DYTerm],
    ) -> Vec<SmtConstraint> {
        let mut constraints = Vec::new();

        for sv in &self.step_vars {
            if !self.config.enable_interception {
                continue;
            }

            let intercept_knowledge = knowledge_encoder.encode_intercept_knowledge(
                message_terms,
                sv.step,
                SmtExpr::var(&sv.intercept_flag),
            );

            for (i, formula) in intercept_knowledge.into_iter().enumerate() {
                constraints.push(SmtConstraint::new(
                    formula,
                    ConstraintOrigin::AdversaryAction {
                        action_idx: sv.step,
                        step: sv.step,
                    },
                    format!("intercept_knowledge_{}_t{}", i, sv.step),
                ));
            }
        }
        constraints
    }

    /// Encode symmetry breaking: adversary actions are ordered.
    pub fn encode_symmetry_breaking(&self) -> Vec<SmtConstraint> {
        if !self.config.enable_symmetry_breaking {
            return Vec::new();
        }

        let mut constraints = Vec::new();

        // Active steps come before inactive steps (among adversary slots)
        for i in 0..self.step_vars.len().saturating_sub(1) {
            let curr = &self.step_vars[i];
            let next = &self.step_vars[i + 1];

            // If next is active, current must be active
            // (or they can both be inactive)
            constraints.push(SmtConstraint::new(
                SmtExpr::implies(
                    SmtExpr::var(&next.active),
                    SmtExpr::var(&curr.active),
                ),
                ConstraintOrigin::SymmetryBreaking,
                format!("symmetry_ordering_{}_{}", i, i + 1),
            ));
        }

        constraints
    }

    /// Encode all adversary constraints.
    pub fn encode_all(&self, depth: u32) -> Vec<SmtConstraint> {
        let mut constraints = Vec::new();
        constraints.push(self.encode_budget_constraint());
        constraints.extend(self.encode_per_action_budgets());
        constraints.extend(self.encode_action_consistency());
        constraints.extend(self.encode_symmetry_breaking());
        constraints
    }

    pub fn step_vars(&self) -> &[StepActionVars] {
        &self.step_vars
    }

    pub fn budget(&self) -> &AdversaryBudget {
        &self.budget
    }

    pub fn config(&self) -> &AdversaryEncoderConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_step_action_vars() {
        let sv = StepActionVars::new(3);
        assert_eq!(sv.step, 3);
        assert_eq!(sv.active, "adv_active_t3");
        assert_eq!(sv.drop_flag, "adv_drop_t3");
    }

    #[test]
    fn test_step_declarations() {
        let sv = StepActionVars::new(0);
        let decls = sv.declarations();
        assert!(decls.len() >= 8);
    }

    #[test]
    fn test_exactly_one_action() {
        let sv = StepActionVars::new(0);
        let config = AdversaryEncoderConfig::default();
        let constraint = sv.exactly_one_action(&config);
        let s = format!("{}", constraint);
        assert!(s.contains("=>"));
    }

    #[test]
    fn test_inactive_means_no_flags() {
        let sv = StepActionVars::new(0);
        let constraint = sv.inactive_means_no_flags();
        let s = format!("{}", constraint);
        assert!(s.contains("not"));
    }

    #[test]
    fn test_adversary_encoder_init() {
        let budget = AdversaryBudget::new(3);
        let mut encoder = AdversaryEncoder::with_default_config(budget);
        encoder.initialize(5);
        assert_eq!(encoder.step_vars().len(), 6);
    }

    #[test]
    fn test_budget_constraint() {
        let budget = AdversaryBudget::new(3);
        let mut encoder = AdversaryEncoder::with_default_config(budget);
        encoder.initialize(5);

        let constraint = encoder.encode_budget_constraint();
        let s = format!("{}", constraint.formula);
        assert!(s.contains("<="));
    }

    #[test]
    fn test_per_action_budgets() {
        let budget = AdversaryBudget::with_per_action_limits(5, 2, 2, 1, 3);
        let mut encoder = AdversaryEncoder::with_default_config(budget);
        encoder.initialize(5);

        let constraints = encoder.encode_per_action_budgets();
        assert!(constraints.len() >= 4); // drop, inject, modify, intercept
    }

    #[test]
    fn test_action_consistency() {
        let budget = AdversaryBudget::new(3);
        let mut encoder = AdversaryEncoder::with_default_config(budget);
        encoder.initialize(3);

        let constraints = encoder.encode_action_consistency();
        // 3 constraints per step (exactly_one, inactive_no_flags, type_consistency)
        assert_eq!(constraints.len(), 4 * 3);
    }

    #[test]
    fn test_symmetry_breaking() {
        let budget = AdversaryBudget::new(3);
        let mut encoder = AdversaryEncoder::with_default_config(budget);
        encoder.initialize(5);

        let constraints = encoder.encode_symmetry_breaking();
        assert_eq!(constraints.len(), 5); // 6 steps - 1 pairs
    }

    #[test]
    fn test_encode_all() {
        let budget = AdversaryBudget::new(3);
        let mut encoder = AdversaryEncoder::with_default_config(budget);
        encoder.initialize(3);

        let constraints = encoder.encode_all(3);
        assert!(!constraints.is_empty());
    }

    #[test]
    fn test_injection_constraints() {
        let budget = AdversaryBudget::new(3);
        let mut encoder = AdversaryEncoder::with_default_config(budget);
        encoder.initialize(2);

        let ke = KnowledgeEncoder::new(3);
        let terms = vec![DYTerm::nonce(1), DYTerm::nonce(2)];
        let constraints = encoder.encode_injection_constraints(&ke, &terms);
        assert!(!constraints.is_empty());
    }

    #[test]
    fn test_modification_constraints() {
        let budget = AdversaryBudget::new(3);
        let mut encoder = AdversaryEncoder::with_default_config(budget);
        encoder.initialize(2);

        let ke = KnowledgeEncoder::new(3);
        let terms = vec![DYTerm::CipherSuiteId(0x002F)];
        let constraints = encoder.encode_modification_constraints(&ke, &terms);
        assert!(!constraints.is_empty());
    }

    #[test]
    fn test_interception_effects() {
        let budget = AdversaryBudget::new(3);
        let mut encoder = AdversaryEncoder::with_default_config(budget);
        encoder.initialize(2);

        let ke = KnowledgeEncoder::new(3);
        let terms = vec![DYTerm::nonce(1)];
        let constraints = encoder.encode_interception_effects(&ke, &terms);
        assert!(!constraints.is_empty());
    }

    #[test]
    fn test_disabled_actions() {
        let budget = AdversaryBudget::new(3);
        let config = AdversaryEncoderConfig {
            enable_injection: false,
            enable_modification: false,
            enable_drop: true,
            enable_interception: true,
            enable_replay: false,
            enable_symmetry_breaking: false,
            max_derivation_depth: 3,
        };
        let mut encoder = AdversaryEncoder::new(budget, config);
        encoder.initialize(2);

        let constraints = encoder.encode_per_action_budgets();
        // Only drop and intercept enabled
        assert_eq!(constraints.len(), 2);
    }

    #[test]
    fn test_action_type_consistency() {
        let sv = StepActionVars::new(0);
        let constraint = sv.action_type_consistency();
        let s = format!("{}", constraint);
        assert!(s.contains("adv_action_t0"));
    }
}
