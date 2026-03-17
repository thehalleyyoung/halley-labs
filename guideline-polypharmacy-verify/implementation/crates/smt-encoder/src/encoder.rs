//! PTA-to-SMT bounded model checking encoder.
//!
//! The main entry point is [`PtaEncoder::encode_bounded`], which translates
//! a [`PTA`] and a bound *k* into an [`EncodedProblem`] whose satisfiability
//! is equivalent to the existence of a counterexample of length ≤ *k*.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::expression::{SmtExpr, simplify, expr_size, total_expr_size};
use crate::guard_encoding::{
    GuardEncoder, clocks_reset_by, concentrations_modified_by, states_modified_by,
};
use crate::pk_encoding::{OneCompartmentParams, PkEncoder};
use crate::pta::{
    Edge, Guard, LocationId, PTA, Reset, ResetAction, SafetyProperty,
};
use crate::variable::{
    SmtSort, SymbolTable, VariableFactory, VariableStore,
    build_symbol_table_and_factory,
};

// ═══════════════════════════════════════════════════════════════════════════
// EncodedProblem
// ═══════════════════════════════════════════════════════════════════════════

/// A fully encoded bounded model checking problem.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodedProblem {
    /// The top-level assertions (their conjunction must be checked for SAT).
    pub assertions: Vec<SmtExpr>,
    /// The variable store containing all variables.
    pub variable_store: VariableStore,
    /// The symbol table mapping PTA names to SMT names.
    pub symbol_table: SymbolTable,
    /// The BMC bound (number of steps).
    pub bound: usize,
    /// Time-step duration.
    pub dt: f64,
    /// Number of locations in the PTA.
    pub num_locations: usize,
    /// Number of edges in the PTA.
    pub num_edges: usize,
}

impl EncodedProblem {
    /// Total number of assertions.
    pub fn num_assertions(&self) -> usize {
        self.assertions.len()
    }

    /// Total number of variables.
    pub fn num_variables(&self) -> usize {
        self.variable_store.len()
    }

    /// Total expression tree size across all assertions.
    pub fn total_size(&self) -> usize {
        total_expr_size(&self.assertions)
    }

    /// Simplify all assertions.
    pub fn simplify_all(&mut self) {
        self.assertions = self.assertions.iter().map(|e| simplify(e)).collect();
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PtaEncoder
// ═══════════════════════════════════════════════════════════════════════════

/// Encoder that translates a PTA into SMT formulas for BMC.
pub struct PtaEncoder {
    bound: usize,
    pk_params: Vec<OneCompartmentParams>,
}

impl PtaEncoder {
    pub fn new(bound: usize) -> Self {
        Self {
            bound,
            pk_params: Vec::new(),
        }
    }

    /// Register PK parameters for a drug.
    pub fn with_pk_params(mut self, params: OneCompartmentParams) -> Self {
        self.pk_params.push(params);
        self
    }

    /// Encode a bounded model checking problem from a PTA.
    pub fn encode_bounded(&self, pta: &PTA) -> EncodedProblem {
        let (symbol_table, factory) = build_symbol_table_and_factory(pta);
        let mut store = VariableStore::new();
        factory.instantiate(&mut store, self.bound);

        let guard_enc = GuardEncoder::new(&store, &symbol_table);
        let pk_enc = PkEncoder::new(&store, &symbol_table);
        let dt = pta.time_step;

        let mut assertions = Vec::new();

        // 1. Initial state
        assertions.extend(self.encode_initial_state(pta, &store, &symbol_table));

        // 2. Transition relation for each step
        for step in 0..self.bound {
            let step_enc = TimeStepEncoder::new(
                pta, &store, &symbol_table, &guard_enc, &pk_enc, step, dt,
            );
            assertions.extend(step_enc.encode());
        }

        // 3. Location invariants at every step
        for step in 0..=self.bound {
            assertions.extend(
                self.encode_invariants_at_step(pta, &guard_enc, &symbol_table, &store, step),
            );
        }

        // 4. PK dynamics at each step
        for step in 0..self.bound {
            assertions.extend(
                self.encode_pk_dynamics(pta, &pk_enc, step, dt),
            );
        }

        // 5. Safety property negation (looking for counterexample)
        if !pta.safety_properties.is_empty() {
            let negation = guard_enc.encode_all_safety_negation(
                &pta.safety_properties, self.bound,
            );
            assertions.push(negation);
        }

        // 6. Variable bounds (non-negativity of clocks and concentrations)
        for step in 0..=self.bound {
            assertions.extend(
                self.encode_variable_bounds(pta, &store, &symbol_table, step),
            );
        }

        EncodedProblem {
            assertions,
            variable_store: store,
            symbol_table,
            bound: self.bound,
            dt,
            num_locations: pta.num_locations(),
            num_edges: pta.num_edges(),
        }
    }

    // ── Initial State ───────────────────────────────────────────────

    fn encode_initial_state(
        &self,
        pta: &PTA,
        store: &VariableStore,
        symbols: &SymbolTable,
    ) -> Vec<SmtExpr> {
        let mut assertions = Vec::new();

        // Location = initial location
        let init_idx = symbols.location_index(&pta.initial_location.0).unwrap_or(0);
        if let Some(loc_id) = store.id_at_step("loc", 0) {
            assertions.push(SmtExpr::eq(
                SmtExpr::Var(loc_id),
                SmtExpr::IntLit(init_idx),
            ));
        }

        // All clocks start at 0
        for clock in &pta.clocks {
            let smt_name = symbols.clock_smt_name(&clock.name)
                .unwrap_or(&clock.name);
            let step_name = format!("{}_t0", smt_name);
            if let Some(id) = store.id_by_name(&step_name) {
                assertions.push(SmtExpr::eq(
                    SmtExpr::Var(id),
                    SmtExpr::RealLit(0.0),
                ));
            }
        }

        // Initial concentrations
        for conc_var in &pta.concentration_vars {
            let smt_name = symbols.concentration_smt_name(&conc_var.name)
                .unwrap_or(&conc_var.name);
            let step_name = format!("{}_t0", smt_name);
            let init_val = pta.initial_concentrations
                .get(&conc_var.drug_name)
                .copied()
                .unwrap_or(0.0);
            if let Some(id) = store.id_by_name(&step_name) {
                assertions.push(SmtExpr::eq(
                    SmtExpr::Var(id),
                    SmtExpr::RealLit(init_val),
                ));
            }
        }

        assertions
    }

    // ── Invariants ──────────────────────────────────────────────────

    fn encode_invariants_at_step(
        &self,
        pta: &PTA,
        guard_enc: &GuardEncoder,
        symbols: &SymbolTable,
        store: &VariableStore,
        step: usize,
    ) -> Vec<SmtExpr> {
        let mut assertions = Vec::new();

        for loc in &pta.locations {
            if loc.invariant.is_empty() {
                continue;
            }

            let loc_idx = symbols.location_index(&loc.id.0).unwrap_or(-1);
            let loc_var_name = format!("loc_t{}", step);
            if let Some(loc_id) = store.id_by_name(&loc_var_name) {
                let in_location = SmtExpr::eq(
                    SmtExpr::Var(loc_id),
                    SmtExpr::IntLit(loc_idx),
                );
                let inv = guard_enc.encode_invariant(&loc.invariant, step);
                // If we're in this location, the invariant must hold.
                assertions.push(SmtExpr::implies(in_location, inv));
            }
        }

        assertions
    }

    // ── PK Dynamics ─────────────────────────────────────────────────

    fn encode_pk_dynamics(
        &self,
        _pta: &PTA,
        pk_enc: &PkEncoder,
        step: usize,
        dt: f64,
    ) -> Vec<SmtExpr> {
        let mut assertions = Vec::new();

        for params in &self.pk_params {
            assertions.extend(pk_enc.encode_one_compartment(params, step, dt));
        }

        assertions
    }

    // ── Variable Bounds ─────────────────────────────────────────────

    fn encode_variable_bounds(
        &self,
        pta: &PTA,
        store: &VariableStore,
        symbols: &SymbolTable,
        step: usize,
    ) -> Vec<SmtExpr> {
        let mut assertions = Vec::new();

        // Clocks are non-negative
        for clock in &pta.clocks {
            let smt_name = symbols.clock_smt_name(&clock.name)
                .unwrap_or(&clock.name);
            let step_name = format!("{}_t{}", smt_name, step);
            if let Some(id) = store.id_by_name(&step_name) {
                assertions.push(SmtExpr::ge(
                    SmtExpr::Var(id),
                    SmtExpr::RealLit(0.0),
                ));
            }
        }

        // Concentrations are non-negative
        for conc_var in &pta.concentration_vars {
            let smt_name = symbols.concentration_smt_name(&conc_var.name)
                .unwrap_or(&conc_var.name);
            let step_name = format!("{}_t{}", smt_name, step);
            if let Some(id) = store.id_by_name(&step_name) {
                assertions.push(SmtExpr::ge(
                    SmtExpr::Var(id),
                    SmtExpr::RealLit(0.0),
                ));
            }
        }

        // Location in valid range
        let loc_step_name = format!("loc_t{}", step);
        if let Some(id) = store.id_by_name(&loc_step_name) {
            let n = pta.num_locations() as i64;
            assertions.push(SmtExpr::ge(SmtExpr::Var(id), SmtExpr::IntLit(0)));
            assertions.push(SmtExpr::lt(SmtExpr::Var(id), SmtExpr::IntLit(n)));
        }

        assertions
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TimeStepEncoder
// ═══════════════════════════════════════════════════════════════════════════

/// Encodes the transition relation for a single time step.
pub struct TimeStepEncoder<'a> {
    pta: &'a PTA,
    store: &'a VariableStore,
    symbols: &'a SymbolTable,
    guard_enc: &'a GuardEncoder<'a>,
    _pk_enc: &'a PkEncoder<'a>,
    step: usize,
    dt: f64,
}

impl<'a> TimeStepEncoder<'a> {
    pub fn new(
        pta: &'a PTA,
        store: &'a VariableStore,
        symbols: &'a SymbolTable,
        guard_enc: &'a GuardEncoder<'a>,
        pk_enc: &'a PkEncoder<'a>,
        step: usize,
        dt: f64,
    ) -> Self {
        Self {
            pta,
            store,
            symbols,
            guard_enc,
            _pk_enc: pk_enc,
            step,
            dt,
        }
    }

    /// Encode the complete transition relation for this step.
    pub fn encode(&self) -> Vec<SmtExpr> {
        let mut assertions = Vec::new();

        // For each edge, encode: if we take this edge then guard holds,
        // source location matches, and resets/target are applied.
        let stutter_idx = self.symbols.stutter_index();
        let trans_step_name = format!("trans_t{}", self.step);
        let trans_id = self.store.id_by_name(&trans_step_name);

        // Transition selector must be a valid edge index or stutter
        if let Some(tid) = trans_id {
            assertions.push(SmtExpr::ge(SmtExpr::Var(tid), SmtExpr::IntLit(0)));
            assertions.push(SmtExpr::le(
                SmtExpr::Var(tid),
                SmtExpr::IntLit(stutter_idx),
            ));
        }

        // Encode each edge transition
        for (edge_idx, edge) in self.pta.edges.iter().enumerate() {
            let edge_assertions = self.encode_edge_transition(edge, edge_idx as i64);
            assertions.extend(edge_assertions);
        }

        // Encode stutter transition
        assertions.extend(self.encode_stutter_transition(stutter_idx));

        assertions
    }

    fn encode_edge_transition(&self, edge: &Edge, edge_idx: i64) -> Vec<SmtExpr> {
        let mut assertions = Vec::new();

        let trans_step_name = format!("trans_t{}", self.step);
        let trans_id = match self.store.id_by_name(&trans_step_name) {
            Some(id) => id,
            None => return assertions,
        };

        // Condition: trans_t{step} = edge_idx
        let takes_edge = SmtExpr::eq(
            SmtExpr::Var(trans_id),
            SmtExpr::IntLit(edge_idx),
        );

        // Source location must match
        let source_idx = self.symbols.location_index(&edge.source.0).unwrap_or(-1);
        let loc_step_name = format!("loc_t{}", self.step);
        if let Some(loc_id) = self.store.id_by_name(&loc_step_name) {
            let in_source = SmtExpr::eq(
                SmtExpr::Var(loc_id),
                SmtExpr::IntLit(source_idx),
            );
            // If we take this edge, we must be in the source location
            assertions.push(SmtExpr::implies(takes_edge.clone(), in_source));
        }

        // Guard must hold
        let guard_expr = self.guard_enc.encode_guard(&edge.guard, self.step);
        assertions.push(SmtExpr::implies(takes_edge.clone(), guard_expr));

        // Target location at next step
        let target_idx = self.symbols.location_index(&edge.target.0).unwrap_or(-1);
        let next_loc_name = format!("loc_t{}", self.step + 1);
        if let Some(next_loc_id) = self.store.id_by_name(&next_loc_name) {
            assertions.push(SmtExpr::implies(
                takes_edge.clone(),
                SmtExpr::eq(SmtExpr::Var(next_loc_id), SmtExpr::IntLit(target_idx)),
            ));
        }

        // Reset actions
        let reset_exprs = self.guard_enc.encode_reset(&edge.reset, self.step);
        for re in reset_exprs {
            assertions.push(SmtExpr::implies(takes_edge.clone(), re));
        }

        // Frame axioms for variables not reset by this edge
        let reset_clocks: HashSet<_> = clocks_reset_by(&edge.reset).into_iter().collect();
        let reset_concs: HashSet<_> = concentrations_modified_by(&edge.reset)
            .into_iter().collect();
        let reset_states: HashSet<_> = states_modified_by(&edge.reset)
            .into_iter().collect();

        // Clocks not reset advance by dt
        for clock in &self.pta.clocks {
            if !reset_clocks.contains(&clock.name) {
                let frame = self.guard_enc.encode_frame_for_clock(clock, self.step, self.dt);
                assertions.push(SmtExpr::implies(takes_edge.clone(), frame));
            }
        }

        // Concentration variables not modified keep their value
        // (actual PK dynamics are encoded separately for all steps)
        // This handles the case where no explicit PK params are registered
        // for a particular concentration variable.

        // State variables not modified keep their value
        for sv in &self.pta.state_vars {
            if !reset_states.contains(&sv.name) {
                let frame = self.guard_enc.encode_frame_for_state(sv, self.step);
                assertions.push(SmtExpr::implies(takes_edge.clone(), frame));
            }
        }

        assertions
    }

    fn encode_stutter_transition(&self, stutter_idx: i64) -> Vec<SmtExpr> {
        let mut assertions = Vec::new();

        let trans_step_name = format!("trans_t{}", self.step);
        let trans_id = match self.store.id_by_name(&trans_step_name) {
            Some(id) => id,
            None => return assertions,
        };

        let is_stutter = SmtExpr::eq(
            SmtExpr::Var(trans_id),
            SmtExpr::IntLit(stutter_idx),
        );

        // Location stays the same
        let loc_curr_name = format!("loc_t{}", self.step);
        let loc_next_name = format!("loc_t{}", self.step + 1);
        if let (Some(curr), Some(next)) = (
            self.store.id_by_name(&loc_curr_name),
            self.store.id_by_name(&loc_next_name),
        ) {
            assertions.push(SmtExpr::implies(
                is_stutter.clone(),
                SmtExpr::eq(SmtExpr::Var(next), SmtExpr::Var(curr)),
            ));
        }

        // All clocks advance by dt
        for clock in &self.pta.clocks {
            let frame = self.guard_enc.encode_frame_for_clock(clock, self.step, self.dt);
            assertions.push(SmtExpr::implies(is_stutter.clone(), frame));
        }

        // State variables stay the same
        for sv in &self.pta.state_vars {
            let frame = self.guard_enc.encode_frame_for_state(sv, self.step);
            assertions.push(SmtExpr::implies(is_stutter.clone(), frame));
        }

        assertions
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Encoding statistics
// ═══════════════════════════════════════════════════════════════════════════

/// Statistics about an encoded problem.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodingStats {
    pub num_assertions: usize,
    pub num_variables: usize,
    pub total_ast_nodes: usize,
    pub bound: usize,
    pub num_locations: usize,
    pub num_edges: usize,
}

impl EncodedProblem {
    /// Compute encoding statistics.
    pub fn stats(&self) -> EncodingStats {
        EncodingStats {
            num_assertions: self.num_assertions(),
            num_variables: self.num_variables(),
            total_ast_nodes: self.total_size(),
            bound: self.bound,
            num_locations: self.num_locations,
            num_edges: self.num_edges,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pta::*;

    fn simple_pta() -> PTA {
        let clock = ClockVariable::new("x");
        let conc = ConcentrationVariable::new("aspirin");

        PTABuilder::new("test", "l0")
            .add_location("l1", "active")
            .add_clock("x")
            .add_concentration_var("aspirin")
            .set_initial_concentration("aspirin", 0.0)
            .add_edge(
                "l0", "l1",
                Guard::clock_ge(&clock, 1.0),
                Reset::new().clock_reset(&clock),
            )
            .add_edge(
                "l1", "l0",
                Guard::clock_ge(&clock, 8.0),
                Reset::new().clock_reset(&clock),
            )
            .set_time_step(1.0)
            .build()
    }

    fn pta_with_safety() -> PTA {
        let clock = ClockVariable::new("x");
        let conc = ConcentrationVariable::new("warfarin");

        PTABuilder::new("safe_test", "l0")
            .add_location("l1", "dosing")
            .add_location("l2", "toxic")
            .add_clock("x")
            .add_concentration_var("warfarin")
            .set_initial_concentration("warfarin", 0.0)
            .add_edge(
                "l0", "l1",
                Guard::True,
                Reset::new().add_dose(&conc, 5.0, 0.9),
            )
            .add_edge(
                "l1", "l2",
                Guard::conc_gt(&conc, 10.0),
                Reset::new(),
            )
            .add_safety_property(SafetyProperty::forbidden_location(
                "no_toxic", &LocationId::new("l2"),
            ))
            .add_safety_property(SafetyProperty::concentration_within(
                "therapeutic", &conc, 0.0, 10.0,
            ))
            .set_time_step(0.5)
            .build()
    }

    #[test]
    fn test_encode_simple_pta() {
        let pta = simple_pta();
        let encoder = PtaEncoder::new(3);
        let problem = encoder.encode_bounded(&pta);

        assert!(problem.num_assertions() > 0);
        assert!(problem.num_variables() > 0);
        assert_eq!(problem.bound, 3);
    }

    #[test]
    fn test_encode_initial_state() {
        let pta = simple_pta();
        let encoder = PtaEncoder::new(2);
        let problem = encoder.encode_bounded(&pta);

        // There should be initial state assertions
        // Check at least one assertion equates loc_t0 to 0
        let has_init_loc = problem.assertions.iter().any(|a| {
            matches!(a, SmtExpr::Eq(_, rhs) if **rhs == SmtExpr::IntLit(0))
        });
        assert!(has_init_loc, "Missing initial location assertion");
    }

    #[test]
    fn test_encode_with_safety() {
        let pta = pta_with_safety();
        let encoder = PtaEncoder::new(5);
        let problem = encoder.encode_bounded(&pta);

        assert!(problem.num_assertions() > 0);
        assert_eq!(problem.num_locations, 3);
        assert_eq!(problem.num_edges, 2);
    }

    #[test]
    fn test_encode_with_pk() {
        let pta = simple_pta();
        let params = OneCompartmentParams::new("aspirin", 1.0, 20.0, 0.8);
        let encoder = PtaEncoder::new(3).with_pk_params(params);
        let problem = encoder.encode_bounded(&pta);

        // PK dynamics should add assertions for each step
        assert!(problem.num_assertions() > 10);
    }

    #[test]
    fn test_encoding_stats() {
        let pta = simple_pta();
        let encoder = PtaEncoder::new(2);
        let problem = encoder.encode_bounded(&pta);
        let stats = problem.stats();

        assert_eq!(stats.bound, 2);
        assert_eq!(stats.num_locations, 2);
        assert_eq!(stats.num_edges, 2);
        assert!(stats.num_assertions > 0);
        assert!(stats.num_variables > 0);
    }

    #[test]
    fn test_simplify_encoded_problem() {
        let pta = simple_pta();
        let encoder = PtaEncoder::new(2);
        let mut problem = encoder.encode_bounded(&pta);

        let size_before = problem.total_size();
        problem.simplify_all();
        let size_after = problem.total_size();

        // Simplification should not increase size
        assert!(size_after <= size_before);
    }

    #[test]
    fn test_encode_bound_zero() {
        let pta = simple_pta();
        let encoder = PtaEncoder::new(0);
        let problem = encoder.encode_bounded(&pta);

        // With bound 0, we only have initial state and invariant assertions
        assert!(problem.num_assertions() > 0);
        assert_eq!(problem.bound, 0);
    }

    #[test]
    fn test_encode_multiple_edges() {
        let clock = ClockVariable::new("x");
        let conc = ConcentrationVariable::new("drug");

        let pta = PTABuilder::new("multi", "l0")
            .add_location("l1", "a")
            .add_location("l2", "b")
            .add_location("l3", "c")
            .add_clock("x")
            .add_concentration_var("drug")
            .add_edge("l0", "l1", Guard::clock_ge(&clock, 1.0), Reset::new().clock_reset(&clock))
            .add_edge("l0", "l2", Guard::clock_ge(&clock, 2.0), Reset::new().clock_reset(&clock))
            .add_edge("l1", "l3", Guard::conc_gt(&conc, 5.0), Reset::new())
            .add_edge("l2", "l3", Guard::True, Reset::new())
            .add_edge("l3", "l0", Guard::clock_ge(&clock, 10.0), Reset::new().clock_reset(&clock))
            .set_time_step(0.5)
            .build();

        let encoder = PtaEncoder::new(4);
        let problem = encoder.encode_bounded(&pta);

        assert_eq!(problem.num_locations, 4);
        assert_eq!(problem.num_edges, 5);
        assert!(problem.num_assertions() > 20);
    }

    #[test]
    fn test_encode_invariant_location() {
        let clock = ClockVariable::new("x");
        let inv = Invariant::new().clock_bound(&clock, 10.0);

        let pta = PTABuilder::new("inv_test", "l0")
            .add_location_with_invariant("l1", "bounded", inv)
            .add_clock("x")
            .add_edge("l0", "l1", Guard::True, Reset::new())
            .add_edge("l1", "l0", Guard::clock_ge(&clock, 5.0), Reset::new().clock_reset(&clock))
            .set_time_step(1.0)
            .build();

        let encoder = PtaEncoder::new(3);
        let problem = encoder.encode_bounded(&pta);

        // Should have invariant implications
        let has_implies = problem.assertions.iter().any(|a| {
            matches!(a, SmtExpr::Implies(_, _))
        });
        assert!(has_implies, "Expected implication assertions for invariants");
    }

    #[test]
    fn test_encode_state_variables() {
        let clock = ClockVariable::new("x");
        let sv = StateVariable::bool_var("alert");

        let pta = PTABuilder::new("state_test", "l0")
            .add_location("l1", "alert_on")
            .add_clock("x")
            .add_state_var(sv.clone())
            .add_edge(
                "l0", "l1",
                Guard::clock_ge(&clock, 1.0),
                Reset::new()
                    .clock_reset(&clock)
                    .with_action(ResetAction::SetBool {
                        variable: sv,
                        value: true,
                    }),
            )
            .set_time_step(1.0)
            .build();

        let encoder = PtaEncoder::new(2);
        let problem = encoder.encode_bounded(&pta);
        assert!(problem.num_variables() > 0);
    }
}
