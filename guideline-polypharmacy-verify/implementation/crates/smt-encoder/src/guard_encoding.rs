//! Guard, invariant, reset, and safety-property encoding.
//!
//! Translates PTA guard expressions, location invariants, edge resets, and
//! safety properties into [`SmtExpr`] formulas that reference time-indexed
//! SMT variables.

use crate::expression::SmtExpr;
use crate::pta::{
    CompoundGuard, Guard, GuardOp, Invariant, InvariantClause,
    Reset, ResetAction, SafetyCondition, SafetyProperty,
    ClockVariable, ConcentrationVariable, StateVariable,
};
use crate::variable::{SymbolTable, VariableId, VariableStore};

// ═══════════════════════════════════════════════════════════════════════════
// GuardEncoder
// ═══════════════════════════════════════════════════════════════════════════

/// Encodes PTA guards, invariants, resets, and safety properties into SMT
/// formulas referencing time-indexed variables.
#[derive(Debug)]
pub struct GuardEncoder<'a> {
    store: &'a VariableStore,
    symbols: &'a SymbolTable,
}

impl<'a> GuardEncoder<'a> {
    pub fn new(store: &'a VariableStore, symbols: &'a SymbolTable) -> Self {
        Self { store, symbols }
    }

    // ── Guard encoding ──────────────────────────────────────────────

    /// Encode a guard at the given time step.
    pub fn encode_guard(&self, guard: &Guard, step: usize) -> SmtExpr {
        match guard {
            Guard::True => SmtExpr::BoolLit(true),

            Guard::Clock { clock, op, value } => {
                self.encode_clock_guard(clock, *op, *value, step)
            }

            Guard::Concentration { variable, op, threshold } => {
                self.encode_concentration_guard(variable, *op, *threshold, step)
            }

            Guard::State { variable, op, value } => {
                self.encode_state_guard(variable, *op, *value, step)
            }

            Guard::BoolState { variable, expected } => {
                self.encode_bool_state_guard(variable, *expected, step)
            }

            Guard::Compound(compound) => self.encode_compound_guard(compound, step),
        }
    }

    fn encode_clock_guard(
        &self,
        clock: &ClockVariable,
        op: GuardOp,
        value: f64,
        step: usize,
    ) -> SmtExpr {
        let smt_name = self.symbols.clock_smt_name(&clock.name)
            .unwrap_or(&clock.name);
        let var_expr = self.var_at_step(smt_name, step);
        let val_expr = SmtExpr::RealLit(value);
        self.comparison(op, var_expr, val_expr)
    }

    fn encode_concentration_guard(
        &self,
        variable: &ConcentrationVariable,
        op: GuardOp,
        threshold: f64,
        step: usize,
    ) -> SmtExpr {
        let smt_name = self.symbols.concentration_smt_name(&variable.name)
            .unwrap_or(&variable.name);
        let var_expr = self.var_at_step(smt_name, step);
        let val_expr = SmtExpr::RealLit(threshold);
        self.comparison(op, var_expr, val_expr)
    }

    fn encode_state_guard(
        &self,
        variable: &StateVariable,
        op: GuardOp,
        value: f64,
        step: usize,
    ) -> SmtExpr {
        let smt_name = self.symbols.state_smt_name(&variable.name)
            .unwrap_or(&variable.name);
        let var_expr = self.var_at_step(smt_name, step);
        let val_expr = SmtExpr::RealLit(value);
        self.comparison(op, var_expr, val_expr)
    }

    fn encode_bool_state_guard(
        &self,
        variable: &StateVariable,
        expected: bool,
        step: usize,
    ) -> SmtExpr {
        let smt_name = self.symbols.state_smt_name(&variable.name)
            .unwrap_or(&variable.name);
        let var_expr = self.var_at_step(smt_name, step);
        if expected {
            var_expr
        } else {
            SmtExpr::not(var_expr)
        }
    }

    fn encode_compound_guard(&self, compound: &CompoundGuard, step: usize) -> SmtExpr {
        match compound {
            CompoundGuard::And(guards) => {
                let encoded: Vec<_> = guards.iter()
                    .map(|g| self.encode_guard(g, step))
                    .collect();
                SmtExpr::and(encoded)
            }
            CompoundGuard::Or(guards) => {
                let encoded: Vec<_> = guards.iter()
                    .map(|g| self.encode_guard(g, step))
                    .collect();
                SmtExpr::or(encoded)
            }
            CompoundGuard::Not(guard) => {
                SmtExpr::not(self.encode_guard(guard, step))
            }
        }
    }

    // ── Invariant encoding ──────────────────────────────────────────

    /// Encode a location invariant at the given time step.
    pub fn encode_invariant(&self, invariant: &Invariant, step: usize) -> SmtExpr {
        if invariant.is_empty() {
            return SmtExpr::BoolLit(true);
        }

        let clauses: Vec<SmtExpr> = invariant.clauses.iter()
            .map(|clause| self.encode_invariant_clause(clause, step))
            .collect();

        if clauses.len() == 1 {
            clauses.into_iter().next().unwrap()
        } else {
            SmtExpr::and(clauses)
        }
    }

    fn encode_invariant_clause(&self, clause: &InvariantClause, step: usize) -> SmtExpr {
        match clause {
            InvariantClause::ClockBound { clock, bound } => {
                let smt_name = self.symbols.clock_smt_name(&clock.name)
                    .unwrap_or(&clock.name);
                let var_expr = self.var_at_step(smt_name, step);
                SmtExpr::le(var_expr, SmtExpr::RealLit(*bound))
            }

            InvariantClause::ConcentrationRange { variable, lower, upper } => {
                let smt_name = self.symbols.concentration_smt_name(&variable.name)
                    .unwrap_or(&variable.name);
                let var_expr = self.var_at_step(smt_name, step);

                let mut constraints = Vec::new();
                if let Some(lo) = lower {
                    constraints.push(SmtExpr::ge(var_expr.clone(), SmtExpr::RealLit(*lo)));
                }
                if let Some(hi) = upper {
                    constraints.push(SmtExpr::le(var_expr, SmtExpr::RealLit(*hi)));
                }

                match constraints.len() {
                    0 => SmtExpr::BoolLit(true),
                    1 => constraints.into_iter().next().unwrap(),
                    _ => SmtExpr::and(constraints),
                }
            }

            InvariantClause::StateCondition { variable, op, value } => {
                let smt_name = self.symbols.state_smt_name(&variable.name)
                    .unwrap_or(&variable.name);
                let var_expr = self.var_at_step(smt_name, step);
                self.comparison(*op, var_expr, SmtExpr::RealLit(*value))
            }

            InvariantClause::BoolCondition { variable, expected } => {
                let smt_name = self.symbols.state_smt_name(&variable.name)
                    .unwrap_or(&variable.name);
                let var_expr = self.var_at_step(smt_name, step);
                if *expected {
                    var_expr
                } else {
                    SmtExpr::not(var_expr)
                }
            }
        }
    }

    // ── Reset encoding ──────────────────────────────────────────────

    /// Encode the reset actions that happen when an edge is taken at `step`.
    /// Returns assertions about the values at `step + 1`.
    pub fn encode_reset(&self, reset: &Reset, step: usize) -> Vec<SmtExpr> {
        let next = step + 1;
        let mut exprs = Vec::new();

        for action in &reset.actions {
            match action {
                ResetAction::ClockReset(clock) => {
                    let smt_name = self.symbols.clock_smt_name(&clock.name)
                        .unwrap_or(&clock.name);
                    let next_var = self.var_at_step(smt_name, next);
                    exprs.push(SmtExpr::eq(next_var, SmtExpr::RealLit(0.0)));
                }

                ResetAction::SetConcentration { variable, value } => {
                    let smt_name = self.symbols.concentration_smt_name(&variable.name)
                        .unwrap_or(&variable.name);
                    let next_var = self.var_at_step(smt_name, next);
                    exprs.push(SmtExpr::eq(next_var, SmtExpr::RealLit(*value)));
                }

                ResetAction::AddDose { variable, dose_mg, bioavailability } => {
                    let smt_name = self.symbols.concentration_smt_name(&variable.name)
                        .unwrap_or(&variable.name);
                    let curr_var = self.var_at_step(smt_name, step);
                    let next_var = self.var_at_step(smt_name, next);
                    let absorbed = dose_mg * bioavailability;
                    exprs.push(SmtExpr::eq(
                        next_var,
                        SmtExpr::add(vec![curr_var, SmtExpr::RealLit(absorbed)]),
                    ));
                }

                ResetAction::SetState { variable, value } => {
                    let smt_name = self.symbols.state_smt_name(&variable.name)
                        .unwrap_or(&variable.name);
                    let next_var = self.var_at_step(smt_name, next);
                    exprs.push(SmtExpr::eq(next_var, SmtExpr::RealLit(*value)));
                }

                ResetAction::SetBool { variable, value } => {
                    let smt_name = self.symbols.state_smt_name(&variable.name)
                        .unwrap_or(&variable.name);
                    let next_var = self.var_at_step(smt_name, next);
                    if *value {
                        exprs.push(next_var);
                    } else {
                        exprs.push(SmtExpr::not(next_var));
                    }
                }
            }
        }

        exprs
    }

    /// Encode the "frame" constraint: a variable that is NOT reset keeps
    /// its value from step to step+1.
    pub fn encode_frame_for_clock(
        &self,
        clock: &ClockVariable,
        step: usize,
        dt: f64,
    ) -> SmtExpr {
        let smt_name = self.symbols.clock_smt_name(&clock.name)
            .unwrap_or(&clock.name);
        let curr = self.var_at_step(smt_name, step);
        let next = self.var_at_step(smt_name, step + 1);
        SmtExpr::eq(next, SmtExpr::add(vec![curr, SmtExpr::RealLit(dt)]))
    }

    /// Encode the frame constraint for a concentration variable (no change).
    pub fn encode_frame_for_concentration(
        &self,
        variable: &ConcentrationVariable,
        step: usize,
    ) -> SmtExpr {
        let smt_name = self.symbols.concentration_smt_name(&variable.name)
            .unwrap_or(&variable.name);
        let curr = self.var_at_step(smt_name, step);
        let next = self.var_at_step(smt_name, step + 1);
        SmtExpr::eq(next, curr)
    }

    /// Encode the frame constraint for a state variable (no change).
    pub fn encode_frame_for_state(&self, variable: &StateVariable, step: usize) -> SmtExpr {
        let smt_name = self.symbols.state_smt_name(&variable.name)
            .unwrap_or(&variable.name);
        let curr = self.var_at_step(smt_name, step);
        let next = self.var_at_step(smt_name, step + 1);
        SmtExpr::eq(next, curr)
    }

    // ── Safety property encoding ────────────────────────────────────

    /// Encode a safety property at a specific step.
    /// Returns an expression that is TRUE when the property HOLDS.
    pub fn encode_safety_property(
        &self,
        property: &SafetyProperty,
        step: usize,
    ) -> SmtExpr {
        self.encode_safety_condition(&property.condition, step)
    }

    fn encode_safety_condition(
        &self,
        condition: &SafetyCondition,
        step: usize,
    ) -> SmtExpr {
        match condition {
            SafetyCondition::ConcentrationBound { variable, lower, upper } => {
                let smt_name = self.symbols.concentration_smt_name(&variable.name)
                    .unwrap_or(&variable.name);
                let var_expr = self.var_at_step(smt_name, step);

                let mut constraints = Vec::new();
                if let Some(lo) = lower {
                    constraints.push(SmtExpr::ge(var_expr.clone(), SmtExpr::RealLit(*lo)));
                }
                if let Some(hi) = upper {
                    constraints.push(SmtExpr::le(var_expr, SmtExpr::RealLit(*hi)));
                }

                match constraints.len() {
                    0 => SmtExpr::BoolLit(true),
                    1 => constraints.into_iter().next().unwrap(),
                    _ => SmtExpr::and(constraints),
                }
            }

            SafetyCondition::ForbiddenLocation(loc_id) => {
                let loc_idx = self.symbols.location_index(&loc_id.0).unwrap_or(-1);
                let loc_var = self.var_at_step("loc", step);
                SmtExpr::not(SmtExpr::eq(loc_var, SmtExpr::IntLit(loc_idx)))
            }

            SafetyCondition::ClockBound { clock, bound } => {
                let smt_name = self.symbols.clock_smt_name(&clock.name)
                    .unwrap_or(&clock.name);
                let var_expr = self.var_at_step(smt_name, step);
                SmtExpr::le(var_expr, SmtExpr::RealLit(*bound))
            }

            SafetyCondition::StateConstraint { variable, op, value } => {
                let smt_name = self.symbols.state_smt_name(&variable.name)
                    .unwrap_or(&variable.name);
                let var_expr = self.var_at_step(smt_name, step);
                self.comparison(*op, var_expr, SmtExpr::RealLit(*value))
            }

            SafetyCondition::BoolMustHold { variable, expected } => {
                let smt_name = self.symbols.state_smt_name(&variable.name)
                    .unwrap_or(&variable.name);
                let var_expr = self.var_at_step(smt_name, step);
                if *expected { var_expr } else { SmtExpr::not(var_expr) }
            }

            SafetyCondition::And(conditions) => {
                let encoded: Vec<_> = conditions.iter()
                    .map(|c| self.encode_safety_condition(c, step))
                    .collect();
                SmtExpr::and(encoded)
            }

            SafetyCondition::Or(conditions) => {
                let encoded: Vec<_> = conditions.iter()
                    .map(|c| self.encode_safety_condition(c, step))
                    .collect();
                SmtExpr::or(encoded)
            }

            SafetyCondition::Not(cond) => {
                SmtExpr::not(self.encode_safety_condition(cond, step))
            }

            SafetyCondition::Implies(ante, cons) => {
                SmtExpr::implies(
                    self.encode_safety_condition(ante, step),
                    self.encode_safety_condition(cons, step),
                )
            }
        }
    }

    /// Encode the NEGATION of a safety property across all steps 0..=bound.
    /// This produces an expression that is satisfiable iff there EXISTS a
    /// step where the safety property is VIOLATED.
    pub fn encode_safety_negation(
        &self,
        property: &SafetyProperty,
        bound: usize,
    ) -> SmtExpr {
        let violations: Vec<SmtExpr> = (0..=bound)
            .map(|step| {
                SmtExpr::not(self.encode_safety_property(property, step))
            })
            .collect();
        SmtExpr::or(violations)
    }

    /// Encode safety negation for multiple properties: exists step, exists
    /// property such that property is violated.
    pub fn encode_all_safety_negation(
        &self,
        properties: &[SafetyProperty],
        bound: usize,
    ) -> SmtExpr {
        let negations: Vec<SmtExpr> = properties.iter()
            .map(|p| self.encode_safety_negation(p, bound))
            .collect();
        SmtExpr::or(negations)
    }

    // ── Helpers ─────────────────────────────────────────────────────

    fn var_at_step(&self, base_name: &str, step: usize) -> SmtExpr {
        let step_name = format!("{}_t{}", base_name, step);
        self.store.id_by_name(&step_name)
            .map(SmtExpr::Var)
            .unwrap_or_else(|| {
                // Fallback: create a named placeholder
                SmtExpr::Apply(step_name, vec![])
            })
    }

    fn comparison(&self, op: GuardOp, lhs: SmtExpr, rhs: SmtExpr) -> SmtExpr {
        match op {
            GuardOp::Lt => SmtExpr::lt(lhs, rhs),
            GuardOp::Le => SmtExpr::le(lhs, rhs),
            GuardOp::Eq => SmtExpr::eq(lhs, rhs),
            GuardOp::Ge => SmtExpr::ge(lhs, rhs),
            GuardOp::Gt => SmtExpr::gt(lhs, rhs),
            GuardOp::Ne => SmtExpr::not(SmtExpr::eq(lhs, rhs)),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Batch guard encoding helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Encode a guard at every step from 0 to bound (inclusive).
pub fn encode_guard_all_steps(
    encoder: &GuardEncoder,
    guard: &Guard,
    bound: usize,
) -> Vec<SmtExpr> {
    (0..=bound).map(|step| encoder.encode_guard(guard, step)).collect()
}

/// Encode an invariant at every step from 0 to bound (inclusive).
pub fn encode_invariant_all_steps(
    encoder: &GuardEncoder,
    invariant: &Invariant,
    bound: usize,
) -> Vec<SmtExpr> {
    (0..=bound).map(|step| encoder.encode_invariant(invariant, step)).collect()
}

/// Determine which clocks are reset by the given Reset.
pub fn clocks_reset_by(reset: &Reset) -> Vec<String> {
    reset.actions.iter().filter_map(|a| {
        match a {
            ResetAction::ClockReset(c) => Some(c.name.clone()),
            _ => None,
        }
    }).collect()
}

/// Determine which concentration variables are modified by the given Reset.
pub fn concentrations_modified_by(reset: &Reset) -> Vec<String> {
    reset.actions.iter().filter_map(|a| {
        match a {
            ResetAction::SetConcentration { variable, .. }
            | ResetAction::AddDose { variable, .. } => Some(variable.name.clone()),
            _ => None,
        }
    }).collect()
}

/// Determine which state variables are modified by the given Reset.
pub fn states_modified_by(reset: &Reset) -> Vec<String> {
    reset.actions.iter().filter_map(|a| {
        match a {
            ResetAction::SetState { variable, .. }
            | ResetAction::SetBool { variable, .. } => Some(variable.name.clone()),
            _ => None,
        }
    }).collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::variable::{SmtSort, VariableStore, SymbolTable};
    use crate::pta::*;

    fn setup() -> (VariableStore, SymbolTable) {
        let mut store = VariableStore::new();
        let mut symbols = SymbolTable::new();

        symbols.register_location("l0", 0);
        symbols.register_location("l1", 1);
        symbols.register_clock("x", "clk_x");
        symbols.register_concentration("conc_warfarin", "conc_warfarin");
        symbols.register_state("active", "sv_active");

        // Create time-indexed variables for steps 0..3
        for step in 0..=3 {
            store.create_time_indexed("loc", SmtSort::Int, step);
            store.create_time_indexed("clk_x", SmtSort::Real, step);
            store.create_time_indexed("conc_warfarin", SmtSort::Real, step);
            store.create_time_indexed("sv_active", SmtSort::Bool, step);
        }

        (store, symbols)
    }

    #[test]
    fn test_encode_true_guard() {
        let (store, symbols) = setup();
        let enc = GuardEncoder::new(&store, &symbols);
        let expr = enc.encode_guard(&Guard::True, 0);
        assert_eq!(expr, SmtExpr::BoolLit(true));
    }

    #[test]
    fn test_encode_clock_guard() {
        let (store, symbols) = setup();
        let enc = GuardEncoder::new(&store, &symbols);
        let clock = ClockVariable::new("x");
        let guard = Guard::clock_ge(&clock, 5.0);
        let expr = enc.encode_guard(&guard, 1);

        // Should be (>= clk_x_t1 5.0)
        match &expr {
            SmtExpr::Ge(_, rhs) => {
                assert_eq!(**rhs, SmtExpr::RealLit(5.0));
            }
            _ => panic!("Expected Ge, got {:?}", expr),
        }
    }

    #[test]
    fn test_encode_concentration_guard() {
        let (store, symbols) = setup();
        let enc = GuardEncoder::new(&store, &symbols);
        let var = ConcentrationVariable::new("warfarin");
        let guard = Guard::conc_lt(&var, 3.0);
        let expr = enc.encode_guard(&guard, 0);

        match &expr {
            SmtExpr::Lt(_, rhs) => {
                assert_eq!(**rhs, SmtExpr::RealLit(3.0));
            }
            _ => panic!("Expected Lt, got {:?}", expr),
        }
    }

    #[test]
    fn test_encode_compound_and_guard() {
        let (store, symbols) = setup();
        let enc = GuardEncoder::new(&store, &symbols);
        let clock = ClockVariable::new("x");
        let guard = Guard::and(vec![
            Guard::clock_ge(&clock, 1.0),
            Guard::clock_le(&clock, 5.0),
        ]);
        let expr = enc.encode_guard(&guard, 0);
        match &expr {
            SmtExpr::And(es) => assert_eq!(es.len(), 2),
            _ => panic!("Expected And"),
        }
    }

    #[test]
    fn test_encode_compound_or_guard() {
        let (store, symbols) = setup();
        let enc = GuardEncoder::new(&store, &symbols);
        let clock = ClockVariable::new("x");
        let guard = Guard::or(vec![
            Guard::clock_lt(&clock, 1.0),
            Guard::clock_gt(&clock, 10.0),
        ]);
        let expr = enc.encode_guard(&guard, 0);
        match &expr {
            SmtExpr::Or(es) => assert_eq!(es.len(), 2),
            _ => panic!("Expected Or"),
        }
    }

    #[test]
    fn test_encode_not_guard() {
        let (store, symbols) = setup();
        let enc = GuardEncoder::new(&store, &symbols);
        let clock = ClockVariable::new("x");
        let guard = Guard::not(Guard::clock_lt(&clock, 1.0));
        let expr = enc.encode_guard(&guard, 0);
        match &expr {
            SmtExpr::Not(_) => {}
            _ => panic!("Expected Not"),
        }
    }

    #[test]
    fn test_encode_empty_invariant() {
        let (store, symbols) = setup();
        let enc = GuardEncoder::new(&store, &symbols);
        let inv = Invariant::new();
        assert_eq!(enc.encode_invariant(&inv, 0), SmtExpr::BoolLit(true));
    }

    #[test]
    fn test_encode_clock_bound_invariant() {
        let (store, symbols) = setup();
        let enc = GuardEncoder::new(&store, &symbols);
        let clock = ClockVariable::new("x");
        let inv = Invariant::new().clock_bound(&clock, 10.0);
        let expr = enc.encode_invariant(&inv, 0);
        match &expr {
            SmtExpr::Le(_, rhs) => assert_eq!(**rhs, SmtExpr::RealLit(10.0)),
            _ => panic!("Expected Le"),
        }
    }

    #[test]
    fn test_encode_concentration_range_invariant() {
        let (store, symbols) = setup();
        let enc = GuardEncoder::new(&store, &symbols);
        let conc = ConcentrationVariable::new("warfarin");
        let inv = Invariant::new().with_clause(InvariantClause::ConcentrationRange {
            variable: conc,
            lower: Some(1.0),
            upper: Some(5.0),
        });
        let expr = enc.encode_invariant(&inv, 0);
        match &expr {
            SmtExpr::And(es) => assert_eq!(es.len(), 2),
            _ => panic!("Expected And"),
        }
    }

    #[test]
    fn test_encode_reset_clock() {
        let (store, symbols) = setup();
        let enc = GuardEncoder::new(&store, &symbols);
        let clock = ClockVariable::new("x");
        let reset = Reset::new().clock_reset(&clock);
        let exprs = enc.encode_reset(&reset, 0);
        assert_eq!(exprs.len(), 1);
        // clk_x_t1 = 0.0
        match &exprs[0] {
            SmtExpr::Eq(_, rhs) => assert_eq!(**rhs, SmtExpr::RealLit(0.0)),
            _ => panic!("Expected Eq"),
        }
    }

    #[test]
    fn test_encode_reset_add_dose() {
        let (store, symbols) = setup();
        let enc = GuardEncoder::new(&store, &symbols);
        let conc = ConcentrationVariable::new("warfarin");
        let reset = Reset::new().add_dose(&conc, 5.0, 0.8);
        let exprs = enc.encode_reset(&reset, 0);
        assert_eq!(exprs.len(), 1);
        // conc_warfarin_t1 = conc_warfarin_t0 + 4.0
        match &exprs[0] {
            SmtExpr::Eq(_, rhs) => {
                match rhs.as_ref() {
                    SmtExpr::Add(terms) => {
                        assert_eq!(terms.len(), 2);
                        assert_eq!(terms[1], SmtExpr::RealLit(4.0));
                    }
                    _ => panic!("Expected Add on rhs"),
                }
            }
            _ => panic!("Expected Eq"),
        }
    }

    #[test]
    fn test_encode_frame_clock() {
        let (store, symbols) = setup();
        let enc = GuardEncoder::new(&store, &symbols);
        let clock = ClockVariable::new("x");
        let expr = enc.encode_frame_for_clock(&clock, 0, 0.5);
        // clk_x_t1 = clk_x_t0 + 0.5
        match &expr {
            SmtExpr::Eq(_, rhs) => {
                match rhs.as_ref() {
                    SmtExpr::Add(terms) => {
                        assert_eq!(terms[1], SmtExpr::RealLit(0.5));
                    }
                    _ => panic!("Expected Add on rhs"),
                }
            }
            _ => panic!("Expected Eq"),
        }
    }

    #[test]
    fn test_encode_frame_concentration() {
        let (store, symbols) = setup();
        let enc = GuardEncoder::new(&store, &symbols);
        let conc = ConcentrationVariable::new("warfarin");
        let expr = enc.encode_frame_for_concentration(&conc, 0);
        // conc_warfarin_t1 = conc_warfarin_t0
        match &expr {
            SmtExpr::Eq(lhs, rhs) => {
                assert_ne!(**lhs, **rhs); // different time steps
            }
            _ => panic!("Expected Eq"),
        }
    }

    #[test]
    fn test_encode_safety_concentration_bound() {
        let (store, symbols) = setup();
        let enc = GuardEncoder::new(&store, &symbols);
        let conc = ConcentrationVariable::new("warfarin");
        let prop = SafetyProperty::concentration_within("safe", &conc, 1.0, 5.0);
        let expr = enc.encode_safety_property(&prop, 0);
        match &expr {
            SmtExpr::And(es) => assert_eq!(es.len(), 2),
            _ => panic!("Expected And"),
        }
    }

    #[test]
    fn test_encode_safety_forbidden_location() {
        let (store, symbols) = setup();
        let enc = GuardEncoder::new(&store, &symbols);
        let prop = SafetyProperty::forbidden_location("no_l1", &LocationId::new("l1"));
        let expr = enc.encode_safety_property(&prop, 0);
        // Should be NOT(loc_t0 = 1)
        match &expr {
            SmtExpr::Not(inner) => {
                match inner.as_ref() {
                    SmtExpr::Eq(_, rhs) => assert_eq!(**rhs, SmtExpr::IntLit(1)),
                    _ => panic!("Expected Eq inside Not"),
                }
            }
            _ => panic!("Expected Not"),
        }
    }

    #[test]
    fn test_encode_safety_negation() {
        let (store, symbols) = setup();
        let enc = GuardEncoder::new(&store, &symbols);
        let conc = ConcentrationVariable::new("warfarin");
        let prop = SafetyProperty::concentration_within("safe", &conc, 1.0, 5.0);
        let expr = enc.encode_safety_negation(&prop, 2);
        // Or of 3 Not(property@step)
        match &expr {
            SmtExpr::Or(es) => {
                assert_eq!(es.len(), 3); // steps 0, 1, 2
                for e in es {
                    assert!(matches!(e, SmtExpr::Not(_)));
                }
            }
            _ => panic!("Expected Or"),
        }
    }

    #[test]
    fn test_clocks_reset_by() {
        let c1 = ClockVariable::new("x");
        let c2 = ClockVariable::new("y");
        let reset = Reset::new().clock_reset(&c1).clock_reset(&c2);
        let clocks = clocks_reset_by(&reset);
        assert_eq!(clocks.len(), 2);
        assert!(clocks.contains(&"x".to_string()));
        assert!(clocks.contains(&"y".to_string()));
    }

    #[test]
    fn test_concentrations_modified_by() {
        let conc = ConcentrationVariable::new("warfarin");
        let reset = Reset::new().add_dose(&conc, 5.0, 0.9);
        let mods = concentrations_modified_by(&reset);
        assert_eq!(mods.len(), 1);
    }

    #[test]
    fn test_encode_bool_state_guard() {
        let (store, symbols) = setup();
        let enc = GuardEncoder::new(&store, &symbols);
        let var = StateVariable::bool_var("active");
        let guard = Guard::bool_guard(&var, true);
        let expr = enc.encode_guard(&guard, 0);
        // Should just be the variable (not wrapped in Not)
        assert!(matches!(expr, SmtExpr::Var(_) | SmtExpr::Apply(_, _)));
    }

    #[test]
    fn test_encode_safety_implies() {
        let (store, symbols) = setup();
        let enc = GuardEncoder::new(&store, &symbols);
        let conc = ConcentrationVariable::new("warfarin");
        let clock = ClockVariable::new("x");
        let prop = SafetyProperty::new(
            "conditional",
            "if clock > 5 then conc in range",
            SafetyCondition::Implies(
                Box::new(SafetyCondition::ClockBound { clock, bound: 5.0 }),
                Box::new(SafetyCondition::ConcentrationBound {
                    variable: conc,
                    lower: Some(1.0),
                    upper: Some(5.0),
                }),
            ),
        );
        let expr = enc.encode_safety_property(&prop, 0);
        assert!(matches!(expr, SmtExpr::Implies(_, _)));
    }
}
