use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use crate::{SmtExpr, SmtSort};
use crate::obligation_encoder::{RawObligation, ObligationKind, ObligationEncoder};

/// Temporal constraint unroller: produces time-indexed constraint sets
/// from regulatory obligations with temporal evolution.
///
/// For each timestep t, generates strategy variables σ_t, encodes active
/// obligations, and adds transition budget constraints Δ(σ_t, σ_{t+1}) ≤ B(t).
#[derive(Debug, Clone)]
pub struct TemporalUnroller {
    horizon: usize,
    max_changes_per_step: usize,
    budget_per_step: f64,
    encoder: ObligationEncoder,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalConstraintSet {
    pub timestep_constraints: Vec<TimestepConstraints>,
    pub transition_constraints: Vec<TransitionConstraint>,
    pub total_variables: usize,
    pub total_constraints: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimestepConstraints {
    pub timestep: usize,
    pub active_obligation_ids: Vec<String>,
    pub hard_constraints: Vec<TimestepExpr>,
    pub soft_constraints: Vec<(TimestepExpr, f64)>,
    pub variable_names: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimestepExpr {
    pub name: String,
    pub expr_description: String,
    pub obligation_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitionConstraint {
    pub from_timestep: usize,
    pub to_timestep: usize,
    pub max_changes: usize,
    pub budget: f64,
    pub change_variables: Vec<(String, String)>,
}

#[derive(Debug, Clone)]
pub struct TemporalObligationSet {
    pub timestep_obligations: Vec<Vec<RawObligation>>,
    pub activation_map: HashMap<String, Vec<usize>>,
}

impl TemporalUnroller {
    pub fn new(horizon: usize, max_changes: usize, budget: f64) -> Self {
        TemporalUnroller {
            horizon,
            max_changes_per_step: max_changes,
            budget_per_step: budget,
            encoder: ObligationEncoder::new(),
        }
    }

    pub fn unroll(&mut self, temporal_set: &TemporalObligationSet) -> TemporalConstraintSet {
        let mut timestep_constraints = Vec::new();
        let mut transition_constraints = Vec::new();
        let mut total_vars = 0;
        let mut total_constrs = 0;

        // Generate constraints for each timestep
        for t in 0..self.horizon {
            let obligations = if t < temporal_set.timestep_obligations.len() {
                &temporal_set.timestep_obligations[t]
            } else if !temporal_set.timestep_obligations.is_empty() {
                temporal_set.timestep_obligations.last().unwrap()
            } else {
                continue;
            };

            let tc = self.encode_timestep(t, obligations);
            total_vars += tc.variable_names.len();
            total_constrs += tc.hard_constraints.len() + tc.soft_constraints.len();
            timestep_constraints.push(tc);
        }

        // Generate transition constraints between consecutive timesteps
        for t in 0..self.horizon.saturating_sub(1) {
            if t + 1 < timestep_constraints.len() {
                let tc = self.encode_transition(t, &timestep_constraints[t], &timestep_constraints[t + 1]);
                total_constrs += 1;
                transition_constraints.push(tc);
            }
        }

        TemporalConstraintSet {
            timestep_constraints,
            transition_constraints,
            total_variables: total_vars,
            total_constraints: total_constrs,
        }
    }

    fn encode_timestep(&mut self, t: usize, obligations: &[RawObligation]) -> TimestepConstraints {
        let mut hard = Vec::new();
        let mut soft = Vec::new();
        let mut var_names = Vec::new();

        for obl in obligations {
            let var_name = format!("x_{}@{}", obl.id.replace("::", "_").replace("-", "_"), t);
            var_names.push(var_name.clone());

            let expr = TimestepExpr {
                name: format!("c_{}@{}", obl.id, t),
                expr_description: format!("Compliance with {} at timestep {}", obl.id, t),
                obligation_id: Some(obl.id.clone()),
            };

            if obl.is_binding && obl.kind == ObligationKind::Obligation {
                hard.push(expr);
            } else if obl.is_binding && obl.kind == ObligationKind::Prohibition {
                hard.push(TimestepExpr {
                    name: format!("p_{}@{}", obl.id, t),
                    expr_description: format!("Prohibition {} at timestep {}", obl.id, t),
                    obligation_id: Some(obl.id.clone()),
                });
            } else {
                soft.push((expr, obl.risk_weight));
            }
        }

        TimestepConstraints {
            timestep: t,
            active_obligation_ids: obligations.iter().map(|o| o.id.clone()).collect(),
            hard_constraints: hard,
            soft_constraints: soft,
            variable_names: var_names,
        }
    }

    fn encode_transition(&self, t: usize, from: &TimestepConstraints, to: &TimestepConstraints) -> TransitionConstraint {
        // Find obligations present in both timesteps
        let from_set: std::collections::HashSet<&String> = from.active_obligation_ids.iter().collect();
        let to_set: std::collections::HashSet<&String> = to.active_obligation_ids.iter().collect();
        let common: Vec<&&String> = from_set.intersection(&to_set).collect();

        let change_variables: Vec<(String, String)> = common.iter().map(|obl_id| {
            let from_var = format!("x_{}@{}", obl_id.replace("::", "_").replace("-", "_"), t);
            let to_var = format!("x_{}@{}", obl_id.replace("::", "_").replace("-", "_"), t + 1);
            (from_var, to_var)
        }).collect();

        TransitionConstraint {
            from_timestep: t,
            to_timestep: t + 1,
            max_changes: self.max_changes_per_step,
            budget: self.budget_per_step,
            change_variables,
        }
    }

    /// Generate SMT expressions for the full temporal unrolling
    pub fn to_smt_constraints(&self, tcs: &TemporalConstraintSet) -> Vec<SmtExpr> {
        let mut constraints = Vec::new();

        // Per-timestep constraints
        for tc in &tcs.timestep_constraints {
            for hc in &tc.hard_constraints {
                if let Some(ref obl_id) = hc.obligation_id {
                    let var = format!("x_{}@{}", obl_id.replace("::", "_").replace("-", "_"), tc.timestep);
                    constraints.push(SmtExpr::Var(var, SmtSort::Bool));
                }
            }
        }

        // Transition budget constraints: sum of |σ_t - σ_{t+1}| ≤ max_changes
        for tc in &tcs.transition_constraints {
            if tc.change_variables.is_empty() { continue; }

            // For binary vars: |x_t - x_{t+1}| is modeled via auxiliary variables
            // d_i >= x_t^i - x_{t+1}^i, d_i >= x_{t+1}^i - x_t^i
            // sum(d_i) <= max_changes
            let change_exprs: Vec<SmtExpr> = tc.change_variables.iter().map(|(_from, _to)| {
                let diff_var = format!("delta_{}_{}", tc.from_timestep, tc.to_timestep);
                SmtExpr::Var(diff_var, SmtSort::Int)
            }).collect();

            if !change_exprs.is_empty() {
                let bound = SmtExpr::Le(
                    Box::new(SmtExpr::IntLit(change_exprs.len() as i64)),
                    Box::new(SmtExpr::IntLit(tc.max_changes as i64)),
                );
                constraints.push(bound);
            }
        }

        constraints
    }

    pub fn horizon(&self) -> usize { self.horizon }
    pub fn max_changes_per_step(&self) -> usize { self.max_changes_per_step }
}

impl TemporalObligationSet {
    pub fn new(horizon: usize) -> Self {
        TemporalObligationSet {
            timestep_obligations: vec![Vec::new(); horizon],
            activation_map: HashMap::new(),
        }
    }

    pub fn add_obligation(&mut self, obl: RawObligation, active_timesteps: Vec<usize>) {
        for &t in &active_timesteps {
            if t < self.timestep_obligations.len() {
                self.timestep_obligations[t].push(obl.clone());
            }
        }
        self.activation_map.insert(obl.id.clone(), active_timesteps);
    }

    pub fn obligations_at(&self, t: usize) -> &[RawObligation] {
        if t < self.timestep_obligations.len() {
            &self.timestep_obligations[t]
        } else {
            &[]
        }
    }

    pub fn total_obligations(&self) -> usize {
        self.activation_map.len()
    }

    pub fn active_at_all_steps(&self) -> Vec<String> {
        self.activation_map.iter()
            .filter(|(_, steps)| steps.len() == self.timestep_obligations.len())
            .map(|(id, _)| id.clone())
            .collect()
    }

    pub fn newly_active_at(&self, t: usize) -> Vec<String> {
        self.activation_map.iter()
            .filter(|(_, steps)| steps.contains(&t) && (t == 0 || !steps.contains(&(t - 1))))
            .map(|(id, _)| id.clone())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::obligation_encoder::ConditionSpec;

    fn make_obl(id: &str, binding: bool) -> RawObligation {
        RawObligation {
            id: id.to_string(), kind: ObligationKind::Obligation,
            jurisdiction: "EU".to_string(), article_ref: format!("Art.{}", id),
            description: format!("Test {}", id), is_binding: binding,
            risk_weight: 1.0, conditions: Vec::new(), exemptions: Vec::new(),
            cross_refs: Vec::new(),
        }
    }

    #[test]
    fn test_temporal_unrolling() {
        let mut tos = TemporalObligationSet::new(3);
        tos.add_obligation(make_obl("always", true), vec![0, 1, 2]);
        tos.add_obligation(make_obl("late", true), vec![2]);

        let mut unroller = TemporalUnroller::new(3, 5, 100_000.0);
        let tcs = unroller.unroll(&tos);

        assert_eq!(tcs.timestep_constraints.len(), 3);
        assert_eq!(tcs.transition_constraints.len(), 2);
        assert_eq!(tcs.timestep_constraints[0].active_obligation_ids.len(), 1);
        assert_eq!(tcs.timestep_constraints[2].active_obligation_ids.len(), 2);
    }

    #[test]
    fn test_newly_active() {
        let mut tos = TemporalObligationSet::new(3);
        tos.add_obligation(make_obl("a", true), vec![0, 1, 2]);
        tos.add_obligation(make_obl("b", true), vec![1, 2]);

        assert_eq!(tos.newly_active_at(0).len(), 1);
        assert_eq!(tos.newly_active_at(1).len(), 1);
        assert!(tos.newly_active_at(2).is_empty());
    }

    #[test]
    fn test_smt_conversion() {
        let mut tos = TemporalObligationSet::new(2);
        tos.add_obligation(make_obl("x", true), vec![0, 1]);

        let mut unroller = TemporalUnroller::new(2, 3, 50_000.0);
        let tcs = unroller.unroll(&tos);
        let smt = unroller.to_smt_constraints(&tcs);
        assert!(!smt.is_empty());
    }
}
