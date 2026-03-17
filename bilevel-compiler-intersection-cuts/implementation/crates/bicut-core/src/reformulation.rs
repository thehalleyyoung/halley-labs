//! Reformulation selection engine.
//!
//! Maps problem signatures to valid reformulation strategies, provides a cost
//! model for strategy comparison, and handles strategy composition rules.

use bicut_types::{
    BilevelProblem, CouplingType, DifficultyClass, LowerLevelType, ProblemSignature,
    ReformulationKind, DEFAULT_TOLERANCE,
};
use log::debug;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A reformulation strategy with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReformulationStrategy {
    pub kind: ReformulationKind,
    pub cost: StrategyCost,
    pub applicable: bool,
    pub justification: String,
    pub prerequisites: Vec<String>,
    pub composition: Option<Vec<ReformulationKind>>,
}

/// Cost model for a reformulation strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyCost {
    pub estimated_vars: usize,
    pub estimated_constraints: usize,
    pub big_m_required: bool,
    pub complementarity_pairs: usize,
    pub numerical_difficulty: f64,
    pub scalability_score: f64,
    pub overall_score: f64,
}

impl StrategyCost {
    /// Lower score is better.
    pub fn compare(&self, other: &StrategyCost) -> std::cmp::Ordering {
        self.overall_score
            .partial_cmp(&other.overall_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Result of the reformulation selection process.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionResult {
    pub recommended: ReformulationStrategy,
    pub alternatives: Vec<ReformulationStrategy>,
    pub signature: ProblemSignature,
    pub reasoning: Vec<String>,
}

// ---------------------------------------------------------------------------
// Selector
// ---------------------------------------------------------------------------

/// Configuration for the reformulation selector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectorConfig {
    pub prefer_strong_duality: bool,
    pub allow_big_m: bool,
    pub max_big_m_vars: usize,
    pub prefer_decomposition: bool,
    pub numerical_tolerance: f64,
}

impl Default for SelectorConfig {
    fn default() -> Self {
        Self {
            prefer_strong_duality: true,
            allow_big_m: true,
            max_big_m_vars: 10000,
            prefer_decomposition: false,
            numerical_tolerance: DEFAULT_TOLERANCE,
        }
    }
}

/// Reformulation selection engine.
pub struct ReformulationSelector {
    config: SelectorConfig,
}

impl ReformulationSelector {
    pub fn new(config: SelectorConfig) -> Self {
        Self { config }
    }

    pub fn with_defaults() -> Self {
        Self::new(SelectorConfig::default())
    }

    /// Select the best reformulation strategy for a given problem.
    pub fn select(&self, problem: &BilevelProblem, sig: &ProblemSignature) -> SelectionResult {
        let mut strategies = Vec::new();
        let mut reasoning = Vec::new();

        // Evaluate each candidate strategy
        let kkt = self.evaluate_kkt(problem, sig);
        strategies.push(kkt);

        let sd = self.evaluate_strong_duality(problem, sig);
        strategies.push(sd);

        let vf = self.evaluate_value_function(problem, sig);
        strategies.push(vf);

        let ccg = self.evaluate_ccg(problem, sig);
        strategies.push(ccg);

        let benders = self.evaluate_benders(problem, sig);
        strategies.push(benders);

        let reg = self.evaluate_regularization(problem, sig);
        strategies.push(reg);

        // Filter to applicable strategies
        let mut applicable: Vec<ReformulationStrategy> =
            strategies.into_iter().filter(|s| s.applicable).collect();

        // Sort by cost (lower overall_score is better)
        applicable.sort_by(|a, b| {
            a.cost
                .overall_score
                .partial_cmp(&b.cost.overall_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Apply preferences
        if self.config.prefer_strong_duality {
            if let Some(pos) = applicable
                .iter()
                .position(|s| s.kind == ReformulationKind::StrongDuality)
            {
                let sd_strategy = applicable[pos].clone();
                if sd_strategy.cost.overall_score
                    < applicable
                        .first()
                        .map(|s| s.cost.overall_score + 0.5)
                        .unwrap_or(f64::MAX)
                {
                    reasoning.push("Preferred StrongDuality due to configuration".to_string());
                }
            }
        }

        if self.config.prefer_decomposition {
            if let Some(pos) = applicable.iter().position(|s| {
                matches!(
                    s.kind,
                    ReformulationKind::BendersDecomposition
                        | ReformulationKind::ColumnConstraintGeneration
                )
            }) {
                reasoning.push("Decomposition preference applied".to_string());
            }
        }

        if !self.config.allow_big_m {
            applicable.retain(|s| !s.cost.big_m_required);
            reasoning.push("Filtered out Big-M strategies per configuration".to_string());
        }

        // Generate overall reasoning
        reasoning.push(format!("Lower-level type: {}", sig.lower_type));
        reasoning.push(format!("Coupling type: {}", sig.coupling_type));
        reasoning.push(format!(
            "Problem size: {} vars, {} constraints",
            sig.total_vars(),
            sig.total_constraints()
        ));

        let recommended = applicable.first().cloned().unwrap_or_else(|| {
            reasoning.push("No applicable strategy; falling back to KKT".to_string());
            self.fallback_strategy(sig)
        });

        let alternatives = if applicable.len() > 1 {
            applicable[1..].to_vec()
        } else {
            Vec::new()
        };

        debug!(
            "Selected reformulation: {} (score: {:.3})",
            recommended.kind, recommended.cost.overall_score
        );

        SelectionResult {
            recommended,
            alternatives,
            signature: sig.clone(),
            reasoning,
        }
    }

    /// Evaluate KKT reformulation.
    fn evaluate_kkt(
        &self,
        problem: &BilevelProblem,
        sig: &ProblemSignature,
    ) -> ReformulationStrategy {
        let n = sig.num_follower_vars;
        let m = sig.num_lower_constraints;

        // KKT is applicable when lower level is convex (LP or QP)
        let applicable = matches!(sig.lower_type, LowerLevelType::LP | LowerLevelType::QP);

        // Cost: adds m dual variables + m complementarity constraints
        let estimated_vars = sig.total_vars() + m;
        let estimated_constraints = sig.total_constraints() + m + m; // stationarity + complementarity
        let complementarity_pairs = m;
        let big_m_required = true; // Need big-M for linearizing complementarity

        // Numerical difficulty increases with problem size and big-M
        let numerical_difficulty = if big_m_required {
            0.4 + 0.1 * (m as f64).log2().max(0.0)
        } else {
            0.2
        };

        // Scalability: KKT adds O(m) variables, moderate scalability
        let scalability = 1.0 - (estimated_vars as f64 / (estimated_vars as f64 + 1000.0));

        let overall = compute_overall_score(
            estimated_vars,
            estimated_constraints,
            big_m_required,
            complementarity_pairs,
            numerical_difficulty,
            scalability,
        );

        ReformulationStrategy {
            kind: ReformulationKind::KKT,
            cost: StrategyCost {
                estimated_vars,
                estimated_constraints,
                big_m_required,
                complementarity_pairs,
                numerical_difficulty,
                scalability_score: scalability,
                overall_score: overall,
            },
            applicable,
            justification: if applicable {
                format!(
                    "KKT applicable: lower level is {}; adds {} duals, {} complementarity pairs",
                    sig.lower_type, m, complementarity_pairs
                )
            } else {
                format!(
                    "KKT not applicable: lower level is {} (non-convex)",
                    sig.lower_type
                )
            },
            prerequisites: if big_m_required {
                vec!["Big-M values must be computed".to_string()]
            } else {
                vec![]
            },
            composition: None,
        }
    }

    /// Evaluate strong duality reformulation.
    fn evaluate_strong_duality(
        &self,
        problem: &BilevelProblem,
        sig: &ProblemSignature,
    ) -> ReformulationStrategy {
        let n = sig.num_follower_vars;
        let m = sig.num_lower_constraints;

        // Strong duality requires LP lower level
        let applicable = sig.lower_type == LowerLevelType::LP;

        let estimated_vars = sig.total_vars() + m; // dual variables
        let estimated_constraints = sig.total_constraints() + n + 1; // dual feasibility + duality gap
        let complementarity_pairs = 0; // no complementarity, just duality gap = 0

        let numerical_difficulty = 0.2 + 0.05 * (n as f64).log2().max(0.0);
        let scalability = 1.0 - (estimated_vars as f64 / (estimated_vars as f64 + 2000.0));

        let overall = compute_overall_score(
            estimated_vars,
            estimated_constraints,
            false,
            complementarity_pairs,
            numerical_difficulty,
            scalability,
        );

        ReformulationStrategy {
            kind: ReformulationKind::StrongDuality,
            cost: StrategyCost {
                estimated_vars,
                estimated_constraints,
                big_m_required: false,
                complementarity_pairs,
                numerical_difficulty,
                scalability_score: scalability,
                overall_score: overall,
            },
            applicable,
            justification: if applicable {
                "Strong duality applicable: LP lower level; no big-M needed".to_string()
            } else {
                format!(
                    "Strong duality not applicable: lower level is {}",
                    sig.lower_type
                )
            },
            prerequisites: vec![
                "Lower level must be bounded".to_string(),
                "Primal-dual feasibility required".to_string(),
            ],
            composition: None,
        }
    }

    /// Evaluate value function reformulation.
    fn evaluate_value_function(
        &self,
        problem: &BilevelProblem,
        sig: &ProblemSignature,
    ) -> ReformulationStrategy {
        let n = sig.num_follower_vars;
        let m = sig.num_lower_constraints;

        // Value function works for LP and QP lower levels
        let applicable = matches!(sig.lower_type, LowerLevelType::LP | LowerLevelType::QP);

        // Value function approach: replaces lower level with value function constraint
        let estimated_vars = sig.total_vars();
        let estimated_constraints = sig.total_constraints() + 1; // + value function constraint

        let numerical_difficulty = 0.3
            + if sig.lower_type == LowerLevelType::QP {
                0.2
            } else {
                0.0
            };
        let scalability = 1.0 - (estimated_vars as f64 / (estimated_vars as f64 + 5000.0));

        let overall = compute_overall_score(
            estimated_vars,
            estimated_constraints,
            false,
            0,
            numerical_difficulty,
            scalability,
        );

        ReformulationStrategy {
            kind: ReformulationKind::ValueFunction,
            cost: StrategyCost {
                estimated_vars,
                estimated_constraints,
                big_m_required: false,
                complementarity_pairs: 0,
                numerical_difficulty,
                scalability_score: scalability,
                overall_score: overall,
            },
            applicable,
            justification: if applicable {
                "Value function: replaces optimality with value function constraint".to_string()
            } else {
                format!(
                    "Value function not applicable for {} lower level",
                    sig.lower_type
                )
            },
            prerequisites: vec![
                "Value function must be computable".to_string(),
                "Lower-level boundedness required".to_string(),
            ],
            composition: None,
        }
    }

    /// Evaluate column-and-constraint generation.
    fn evaluate_ccg(
        &self,
        problem: &BilevelProblem,
        sig: &ProblemSignature,
    ) -> ReformulationStrategy {
        let n_leader = sig.num_leader_vars;
        let n_follower = sig.num_follower_vars;
        let m = sig.num_lower_constraints;

        // CCG works best for LP/MILP lower levels
        let applicable = matches!(
            sig.lower_type,
            LowerLevelType::LP | LowerLevelType::MILP | LowerLevelType::QP
        );

        // CCG: iterative, so initial reformulation is small
        let estimated_vars = n_leader + n_follower;
        let estimated_constraints = sig.num_upper_constraints + 1;

        let numerical_difficulty = 0.15;
        let scalability = 0.9; // Very scalable

        let overall = compute_overall_score(
            estimated_vars,
            estimated_constraints,
            false,
            0,
            numerical_difficulty,
            scalability,
        );

        // Bonus for large problems where CCG shines
        let size_bonus = if sig.total_vars() > 100 { -0.2 } else { 0.0 };
        let adjusted_overall = (overall + size_bonus).max(0.01);

        ReformulationStrategy {
            kind: ReformulationKind::ColumnConstraintGeneration,
            cost: StrategyCost {
                estimated_vars,
                estimated_constraints,
                big_m_required: false,
                complementarity_pairs: 0,
                numerical_difficulty,
                scalability_score: scalability,
                overall_score: adjusted_overall,
            },
            applicable,
            justification: if applicable {
                "CCG: iterative decomposition, good scalability for large instances".to_string()
            } else {
                "CCG not applicable for this lower-level type".to_string()
            },
            prerequisites: vec!["Subproblem solver required".to_string()],
            composition: None,
        }
    }

    /// Evaluate Benders decomposition.
    fn evaluate_benders(
        &self,
        problem: &BilevelProblem,
        sig: &ProblemSignature,
    ) -> ReformulationStrategy {
        // Benders requires LP lower level and works best when follower is much larger than leader
        let applicable =
            sig.lower_type == LowerLevelType::LP && sig.num_follower_vars >= sig.num_leader_vars;

        let estimated_vars = sig.num_leader_vars;
        let estimated_constraints = sig.num_upper_constraints + 1;

        let ratio = if sig.num_leader_vars > 0 {
            sig.num_follower_vars as f64 / sig.num_leader_vars as f64
        } else {
            1.0
        };
        let numerical_difficulty = 0.2;
        let scalability = 0.85;

        let overall = compute_overall_score(
            estimated_vars,
            estimated_constraints,
            false,
            0,
            numerical_difficulty,
            scalability,
        );

        // Bonus when follower >> leader
        let ratio_bonus = if ratio > 5.0 {
            -0.3
        } else if ratio > 2.0 {
            -0.1
        } else {
            0.1
        };
        let adjusted = (overall + ratio_bonus).max(0.01);

        ReformulationStrategy {
            kind: ReformulationKind::BendersDecomposition,
            cost: StrategyCost {
                estimated_vars,
                estimated_constraints,
                big_m_required: false,
                complementarity_pairs: 0,
                numerical_difficulty,
                scalability_score: scalability,
                overall_score: adjusted,
            },
            applicable,
            justification: if applicable {
                format!(
                    "Benders: follower/leader ratio = {:.1}, good decomposition candidate",
                    ratio
                )
            } else {
                "Benders not applicable (requires LP lower level, follower ≥ leader)".to_string()
            },
            prerequisites: vec![
                "Benders master problem solver required".to_string(),
                "Cut generation callback required".to_string(),
            ],
            composition: None,
        }
    }

    /// Evaluate regularization approach.
    fn evaluate_regularization(
        &self,
        problem: &BilevelProblem,
        sig: &ProblemSignature,
    ) -> ReformulationStrategy {
        // Regularization works for any problem type but provides approximate solutions
        let applicable = true;

        let estimated_vars = sig.total_vars();
        let estimated_constraints = sig.total_constraints() + sig.num_follower_vars;

        let numerical_difficulty = 0.5;
        let scalability = 0.7;

        let overall = compute_overall_score(
            estimated_vars,
            estimated_constraints,
            false,
            0,
            numerical_difficulty,
            scalability,
        );

        // Penalize for being approximate
        let adjusted = overall + 0.3;

        ReformulationStrategy {
            kind: ReformulationKind::Regularization,
            cost: StrategyCost {
                estimated_vars,
                estimated_constraints,
                big_m_required: false,
                complementarity_pairs: 0,
                numerical_difficulty,
                scalability_score: scalability,
                overall_score: adjusted,
            },
            applicable,
            justification: "Regularization: applicable to all types, provides approximate solution"
                .to_string(),
            prerequisites: vec!["Regularization parameter selection needed".to_string()],
            composition: None,
        }
    }

    /// Fallback strategy when nothing else applies.
    fn fallback_strategy(&self, sig: &ProblemSignature) -> ReformulationStrategy {
        ReformulationStrategy {
            kind: ReformulationKind::KKT,
            cost: StrategyCost {
                estimated_vars: sig.total_vars() + sig.num_lower_constraints,
                estimated_constraints: sig.total_constraints() * 2,
                big_m_required: true,
                complementarity_pairs: sig.num_lower_constraints,
                numerical_difficulty: 0.8,
                scalability_score: 0.3,
                overall_score: 2.0,
            },
            applicable: true,
            justification: "Fallback: KKT with big-M (may be numerically challenging)".to_string(),
            prerequisites: vec!["Big-M computation required".to_string()],
            composition: None,
        }
    }

    /// Check if two strategies can be composed (e.g., KKT + cuts).
    pub fn can_compose(a: ReformulationKind, b: ReformulationKind) -> bool {
        matches!(
            (a, b),
            (ReformulationKind::KKT, ReformulationKind::Regularization)
                | (
                    ReformulationKind::StrongDuality,
                    ReformulationKind::Regularization
                )
                | (
                    ReformulationKind::ValueFunction,
                    ReformulationKind::ColumnConstraintGeneration
                )
                | (
                    ReformulationKind::BendersDecomposition,
                    ReformulationKind::Regularization
                )
        )
    }

    /// Build a composed strategy from two compatible strategies.
    pub fn compose(
        &self,
        primary: &ReformulationStrategy,
        secondary: &ReformulationStrategy,
    ) -> Option<ReformulationStrategy> {
        if !Self::can_compose(primary.kind, secondary.kind) {
            return None;
        }

        let cost = StrategyCost {
            estimated_vars: primary
                .cost
                .estimated_vars
                .max(secondary.cost.estimated_vars),
            estimated_constraints: primary.cost.estimated_constraints
                + secondary.cost.estimated_constraints / 2,
            big_m_required: primary.cost.big_m_required || secondary.cost.big_m_required,
            complementarity_pairs: primary.cost.complementarity_pairs,
            numerical_difficulty: (primary.cost.numerical_difficulty
                + secondary.cost.numerical_difficulty)
                / 2.0,
            scalability_score: primary
                .cost
                .scalability_score
                .min(secondary.cost.scalability_score),
            overall_score: (primary.cost.overall_score + secondary.cost.overall_score) / 2.0,
        };

        Some(ReformulationStrategy {
            kind: primary.kind,
            cost,
            applicable: true,
            justification: format!("Composed: {} + {}", primary.kind, secondary.kind),
            prerequisites: primary
                .prerequisites
                .iter()
                .chain(secondary.prerequisites.iter())
                .cloned()
                .collect(),
            composition: Some(vec![primary.kind, secondary.kind]),
        })
    }
}

// ---------------------------------------------------------------------------
// Cost model helpers
// ---------------------------------------------------------------------------

fn compute_overall_score(
    vars: usize,
    constraints: usize,
    big_m: bool,
    complementarity: usize,
    numerical_difficulty: f64,
    scalability: f64,
) -> f64 {
    let size_score = (vars as f64 + constraints as f64).log2().max(1.0) / 20.0;
    let big_m_penalty = if big_m { 0.3 } else { 0.0 };
    let comp_penalty = complementarity as f64 * 0.01;
    let difficulty_weight = 0.4;
    let scalability_weight = 0.2;

    size_score
        + big_m_penalty
        + comp_penalty
        + difficulty_weight * numerical_difficulty
        + scalability_weight * (1.0 - scalability)
}

/// Map a problem signature to the set of valid reformulation kinds.
pub fn valid_reformulations(sig: &ProblemSignature) -> Vec<ReformulationKind> {
    let mut kinds = Vec::new();

    // KKT: convex lower level
    if matches!(
        sig.lower_type,
        LowerLevelType::LP | LowerLevelType::QP | LowerLevelType::ConvexNLP
    ) {
        kinds.push(ReformulationKind::KKT);
    }

    // Strong duality: LP lower level only
    if sig.lower_type == LowerLevelType::LP {
        kinds.push(ReformulationKind::StrongDuality);
    }

    // Value function: LP or QP
    if matches!(sig.lower_type, LowerLevelType::LP | LowerLevelType::QP) {
        kinds.push(ReformulationKind::ValueFunction);
    }

    // CCG: LP, QP, MILP
    if matches!(
        sig.lower_type,
        LowerLevelType::LP | LowerLevelType::QP | LowerLevelType::MILP
    ) {
        kinds.push(ReformulationKind::ColumnConstraintGeneration);
    }

    // Benders: LP only
    if sig.lower_type == LowerLevelType::LP {
        kinds.push(ReformulationKind::BendersDecomposition);
    }

    // Regularization: always available
    kinds.push(ReformulationKind::Regularization);

    kinds
}

/// Quick recommendation based only on the signature (no cost model).
pub fn quick_recommend(sig: &ProblemSignature) -> ReformulationKind {
    match sig.lower_type {
        LowerLevelType::LP => {
            if sig.total_vars() > 200 {
                ReformulationKind::ColumnConstraintGeneration
            } else {
                ReformulationKind::StrongDuality
            }
        }
        LowerLevelType::QP => ReformulationKind::KKT,
        LowerLevelType::MILP | LowerLevelType::MIQP => {
            ReformulationKind::ColumnConstraintGeneration
        }
        LowerLevelType::ConvexNLP => ReformulationKind::KKT,
        LowerLevelType::GeneralNLP => ReformulationKind::Regularization,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use bicut_types::{BilevelProblem, SparseMatrix};

    fn make_lp_signature() -> ProblemSignature {
        ProblemSignature {
            lower_type: LowerLevelType::LP,
            coupling_type: CouplingType::Both,
            num_leader_vars: 3,
            num_follower_vars: 5,
            num_upper_constraints: 2,
            num_lower_constraints: 4,
            num_coupling_constraints: 1,
            has_integer_upper: false,
            has_integer_lower: false,
        }
    }

    fn make_milp_signature() -> ProblemSignature {
        ProblemSignature {
            lower_type: LowerLevelType::MILP,
            coupling_type: CouplingType::Both,
            num_leader_vars: 3,
            num_follower_vars: 5,
            num_upper_constraints: 2,
            num_lower_constraints: 4,
            num_coupling_constraints: 1,
            has_integer_upper: false,
            has_integer_lower: true,
        }
    }

    fn make_test_problem() -> BilevelProblem {
        let mut lower_a = SparseMatrix::new(2, 3);
        lower_a.add_entry(0, 0, 1.0);
        lower_a.add_entry(0, 1, 1.0);
        lower_a.add_entry(1, 1, 1.0);
        lower_a.add_entry(1, 2, 1.0);

        let mut linking = SparseMatrix::new(2, 2);
        linking.add_entry(0, 0, 1.0);

        BilevelProblem {
            upper_obj_c_x: vec![1.0, 0.0],
            upper_obj_c_y: vec![0.0, 1.0, 0.0],
            lower_obj_c: vec![1.0, 1.0, 1.0],
            lower_a,
            lower_b: vec![5.0, 5.0],
            lower_linking_b: linking,
            upper_constraints_a: SparseMatrix::new(1, 5),
            upper_constraints_b: vec![10.0],
            num_upper_vars: 2,
            num_lower_vars: 3,
            num_lower_constraints: 2,
            num_upper_constraints: 1,
        }
    }

    #[test]
    fn test_valid_reformulations_lp() {
        let sig = make_lp_signature();
        let valid = valid_reformulations(&sig);
        assert!(valid.contains(&ReformulationKind::KKT));
        assert!(valid.contains(&ReformulationKind::StrongDuality));
        assert!(valid.contains(&ReformulationKind::ValueFunction));
    }

    #[test]
    fn test_valid_reformulations_milp() {
        let sig = make_milp_signature();
        let valid = valid_reformulations(&sig);
        assert!(!valid.contains(&ReformulationKind::KKT));
        assert!(valid.contains(&ReformulationKind::ColumnConstraintGeneration));
    }

    #[test]
    fn test_quick_recommend_lp_small() {
        let mut sig = make_lp_signature();
        sig.num_leader_vars = 5;
        sig.num_follower_vars = 10;
        let rec = quick_recommend(&sig);
        assert_eq!(rec, ReformulationKind::StrongDuality);
    }

    #[test]
    fn test_quick_recommend_lp_large() {
        let mut sig = make_lp_signature();
        sig.num_leader_vars = 100;
        sig.num_follower_vars = 200;
        let rec = quick_recommend(&sig);
        assert_eq!(rec, ReformulationKind::ColumnConstraintGeneration);
    }

    #[test]
    fn test_selector_lp_problem() {
        let selector = ReformulationSelector::with_defaults();
        let sig = make_lp_signature();
        let p = make_test_problem();
        let result = selector.select(&p, &sig);
        assert!(result.recommended.applicable);
        assert!(!result.reasoning.is_empty());
    }

    #[test]
    fn test_no_big_m_config() {
        let selector = ReformulationSelector::new(SelectorConfig {
            allow_big_m: false,
            ..Default::default()
        });
        let sig = make_lp_signature();
        let p = make_test_problem();
        let result = selector.select(&p, &sig);
        assert!(!result.recommended.cost.big_m_required);
    }

    #[test]
    fn test_strategy_composition() {
        let selector = ReformulationSelector::with_defaults();
        let sig = make_lp_signature();
        let p = make_test_problem();
        let result = selector.select(&p, &sig);

        // Try composing KKT + Regularization
        let kkt = ReformulationStrategy {
            kind: ReformulationKind::KKT,
            cost: StrategyCost {
                estimated_vars: 10,
                estimated_constraints: 10,
                big_m_required: true,
                complementarity_pairs: 4,
                numerical_difficulty: 0.4,
                scalability_score: 0.6,
                overall_score: 1.0,
            },
            applicable: true,
            justification: "test".to_string(),
            prerequisites: vec![],
            composition: None,
        };
        let reg = ReformulationStrategy {
            kind: ReformulationKind::Regularization,
            cost: StrategyCost {
                estimated_vars: 8,
                estimated_constraints: 8,
                big_m_required: false,
                complementarity_pairs: 0,
                numerical_difficulty: 0.5,
                scalability_score: 0.7,
                overall_score: 1.2,
            },
            applicable: true,
            justification: "test".to_string(),
            prerequisites: vec![],
            composition: None,
        };
        let composed = selector.compose(&kkt, &reg);
        assert!(composed.is_some());
    }

    #[test]
    fn test_cost_comparison() {
        let c1 = StrategyCost {
            estimated_vars: 10,
            estimated_constraints: 10,
            big_m_required: false,
            complementarity_pairs: 0,
            numerical_difficulty: 0.2,
            scalability_score: 0.8,
            overall_score: 0.5,
        };
        let c2 = StrategyCost {
            estimated_vars: 20,
            estimated_constraints: 20,
            big_m_required: true,
            complementarity_pairs: 4,
            numerical_difficulty: 0.6,
            scalability_score: 0.4,
            overall_score: 1.5,
        };
        assert_eq!(c1.compare(&c2), std::cmp::Ordering::Less);
    }

    #[test]
    fn test_can_compose() {
        assert!(ReformulationSelector::can_compose(
            ReformulationKind::KKT,
            ReformulationKind::Regularization
        ));
        assert!(!ReformulationSelector::can_compose(
            ReformulationKind::KKT,
            ReformulationKind::StrongDuality
        ));
    }

    #[test]
    fn test_fallback_strategy() {
        let sig = ProblemSignature {
            lower_type: LowerLevelType::GeneralNLP,
            coupling_type: CouplingType::Both,
            num_leader_vars: 10,
            num_follower_vars: 10,
            num_upper_constraints: 5,
            num_lower_constraints: 5,
            num_coupling_constraints: 2,
            has_integer_upper: false,
            has_integer_lower: false,
        };
        let rec = quick_recommend(&sig);
        assert_eq!(rec, ReformulationKind::Regularization);
    }
}
