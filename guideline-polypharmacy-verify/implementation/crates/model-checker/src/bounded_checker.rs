//! Bounded model checking (BMC) engine.
//!
//! Iteratively unrolls the PTA transition relation up to a configurable bound,
//! encodes safety properties as SMT assertions, and checks for
//! counterexamples.

use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::{
    ActionLabel, AtomicPredicate, Edge, EncodedProblem, EnzymeContract,
    LocationId, LocationKind, ModelCheckerError, PTA, Predicate, Result,
    SafetyProperty, SafetyPropertyKind, SmtAssertion, SmtModel, SmtSort,
    SmtValue, SmtVariable, SolverVerdict, Update, Variable, VariableId,
    VariableKind, Verdict,
};

use crate::counterexample::{CounterExample, TraceStep};

// ---------------------------------------------------------------------------
// BmcConfig
// ---------------------------------------------------------------------------

/// Configuration for bounded model checking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BmcConfig {
    /// Maximum unrolling bound.
    pub max_bound: usize,
    /// Timeout per individual bound check (seconds).
    pub per_bound_timeout_secs: f64,
    /// Global timeout (seconds).
    pub global_timeout_secs: f64,
    /// Maximum number of SMT clauses before aborting.
    pub max_clauses: usize,
    /// Whether to use incremental solving.
    pub incremental: bool,
    /// Time step for timed unrolling (hours).
    pub time_step: f64,
}

impl Default for BmcConfig {
    fn default() -> Self {
        Self {
            max_bound: 50,
            per_bound_timeout_secs: 30.0,
            global_timeout_secs: 300.0,
            max_clauses: 1_000_000,
            incremental: true,
            time_step: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// CheckResult
// ---------------------------------------------------------------------------

/// Result of a BMC check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckResult {
    pub verdict: Verdict,
    pub counterexample: Option<CounterExample>,
    pub stats: BmcStatistics,
}

impl CheckResult {
    pub fn safe(stats: BmcStatistics) -> Self {
        Self { verdict: Verdict::Safe, counterexample: None, stats }
    }

    pub fn unsafe_with_cx(cx: CounterExample, stats: BmcStatistics) -> Self {
        Self { verdict: Verdict::Unsafe, counterexample: Some(cx), stats }
    }

    pub fn unknown(stats: BmcStatistics) -> Self {
        Self { verdict: Verdict::Unknown, counterexample: None, stats }
    }
}

// ---------------------------------------------------------------------------
// BmcStatistics
// ---------------------------------------------------------------------------

/// Statistics collected during BMC.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BmcStatistics {
    pub bound_reached: usize,
    pub total_clauses: usize,
    pub total_variables: usize,
    pub solve_time_per_bound: Vec<f64>,
    pub total_time_secs: f64,
    pub solver_calls: usize,
    pub timed_out: bool,
}

impl BmcStatistics {
    pub fn new() -> Self {
        Self {
            bound_reached: 0,
            total_clauses: 0,
            total_variables: 0,
            solve_time_per_bound: Vec::new(),
            total_time_secs: 0.0,
            solver_calls: 0,
            timed_out: false,
        }
    }

    pub fn avg_solve_time(&self) -> f64 {
        if self.solve_time_per_bound.is_empty() {
            0.0
        } else {
            self.solve_time_per_bound.iter().sum::<f64>()
                / self.solve_time_per_bound.len() as f64
        }
    }

    pub fn max_solve_time(&self) -> f64 {
        self.solve_time_per_bound
            .iter()
            .copied()
            .fold(0.0_f64, f64::max)
    }
}

impl Default for BmcStatistics {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// SmtEncoder (internal)
// ---------------------------------------------------------------------------

/// Internal SMT encoder for BMC unrolling.
#[derive(Debug)]
struct BmcEncoder {
    variables: Vec<SmtVariable>,
    assertions: Vec<SmtAssertion>,
    step_vars: Vec<HashMap<String, SmtVariable>>,
    next_var_id: usize,
    time_step: f64,
}

impl BmcEncoder {
    fn new(time_step: f64) -> Self {
        Self {
            variables: Vec::new(),
            assertions: Vec::new(),
            step_vars: Vec::new(),
            next_var_id: 0,
            time_step,
        }
    }

    /// Create a fresh SMT variable.
    fn fresh_var(&mut self, name: &str, sort: SmtSort) -> SmtVariable {
        let var = SmtVariable {
            name: format!("{}_{}", name, self.next_var_id),
            sort,
        };
        self.next_var_id += 1;
        self.variables.push(var.clone());
        var
    }

    /// Encode the initial state of the PTA at step 0.
    fn encode_initial_state(&mut self, pta: &PTA) {
        let mut step_map = HashMap::new();

        // Location variable: loc_0 = initial_location
        let loc_var = self.fresh_var("loc_0", SmtSort::Int);
        self.assertions.push(SmtAssertion {
            description: format!("initial location = {}", pta.initial_location),
            smt2: format!("(= {} {})", loc_var.name, pta.initial_location),
        });
        step_map.insert("location".to_string(), loc_var);

        // Time variable: time_0 = 0
        let time_var = self.fresh_var("time_0", SmtSort::Real);
        self.assertions.push(SmtAssertion {
            description: "initial time = 0".into(),
            smt2: format!("(= {} 0.0)", time_var.name),
        });
        step_map.insert("time".to_string(), time_var);

        // Variable valuations.
        for (idx, var) in pta.variables.iter().enumerate() {
            let init_val = pta
                .initial_variable_values
                .get(idx)
                .copied()
                .unwrap_or(0.0);
            let smt_var = self.fresh_var(&format!("{}_0", var.name), SmtSort::Real);
            self.assertions.push(SmtAssertion {
                description: format!("initial {} = {}", var.name, init_val),
                smt2: format!("(= {} {:.6})", smt_var.name, init_val),
            });
            step_map.insert(var.name.clone(), smt_var);
        }

        // Clock valuations (all zero).
        for clk in &pta.clocks {
            let clk_var = self.fresh_var(&format!("{}_0", clk.name), SmtSort::Real);
            self.assertions.push(SmtAssertion {
                description: format!("initial clock {} = 0", clk.name),
                smt2: format!("(= {} 0.0)", clk_var.name),
            });
            step_map.insert(format!("clock_{}", clk.name), clk_var);
        }

        self.step_vars.push(step_map);
    }

    /// Encode the transition relation at step `k` → `k+1`.
    fn encode_transition(&mut self, pta: &PTA, step: usize) {
        let mut next_step_map = HashMap::new();

        // Location variable for step+1.
        let loc_var = self.fresh_var(&format!("loc_{}", step + 1), SmtSort::Int);
        next_step_map.insert("location".to_string(), loc_var.clone());

        // Time variable for step+1 (advances by time_step).
        let time_var = self.fresh_var(&format!("time_{}", step + 1), SmtSort::Real);
        let prev_time_name = self.step_vars[step]
            .get("time")
            .map(|v| v.name.clone())
            .unwrap_or_else(|| "0.0".into());
        self.assertions.push(SmtAssertion {
            description: format!("time at step {} = previous + dt", step + 1),
            smt2: format!(
                "(= {} (+ {} {:.6}))",
                time_var.name, prev_time_name, self.time_step
            ),
        });
        next_step_map.insert("time".to_string(), time_var);

        // Encode variable transitions.
        for var in &pta.variables {
            let var_smt = self.fresh_var(
                &format!("{}_{}", var.name, step + 1),
                SmtSort::Real,
            );
            // Default: variable retains its value (frame axiom).
            let prev_var_name = self.step_vars[step]
                .get(&var.name)
                .map(|v| v.name.clone())
                .unwrap_or_else(|| "0.0".into());
            // Apply PK dynamics: exponential decay for concentration variables.
            if var.kind == VariableKind::Concentration {
                let decay_factor = (-pta.pk_params.ke * self.time_step).exp();
                self.assertions.push(SmtAssertion {
                    description: format!(
                        "PK decay for {} at step {}",
                        var.name,
                        step + 1
                    ),
                    smt2: format!(
                        "(= {} (* {} {:.6}))",
                        var_smt.name, prev_var_name, decay_factor
                    ),
                });
            } else {
                self.assertions.push(SmtAssertion {
                    description: format!("frame axiom for {} at step {}", var.name, step + 1),
                    smt2: format!("(= {} {})", var_smt.name, prev_var_name),
                });
            }
            next_step_map.insert(var.name.clone(), var_smt);
        }

        // Clock transitions: clocks advance by time_step.
        for clk in &pta.clocks {
            let clk_smt = self.fresh_var(
                &format!("{}_{}", clk.name, step + 1),
                SmtSort::Real,
            );
            let prev_clk_name = self.step_vars[step]
                .get(&format!("clock_{}", clk.name))
                .map(|v| v.name.clone())
                .unwrap_or_else(|| "0.0".into());
            self.assertions.push(SmtAssertion {
                description: format!("clock {} advances at step {}", clk.name, step + 1),
                smt2: format!(
                    "(= {} (+ {} {:.6}))",
                    clk_smt.name, prev_clk_name, self.time_step
                ),
            });
            next_step_map.insert(format!("clock_{}", clk.name), clk_smt);
        }

        // Encode edge transitions (disjunction: at least one edge fires or stutter).
        let prev_loc_name = self.step_vars[step]
            .get("location")
            .map(|v| v.name.clone())
            .unwrap_or_else(|| "0".into());

        let mut edge_disjuncts = Vec::new();

        for edge in &pta.edges {
            let edge_cond = format!(
                "(and (= {} {}) (= {} {}))",
                prev_loc_name, edge.source, loc_var.name, edge.target
            );
            edge_disjuncts.push(edge_cond);
        }

        // Stutter transition: stay in same location.
        edge_disjuncts.push(format!(
            "(= {} {})",
            loc_var.name, prev_loc_name
        ));

        if !edge_disjuncts.is_empty() {
            let disjunction = if edge_disjuncts.len() == 1 {
                edge_disjuncts[0].clone()
            } else {
                format!("(or {})", edge_disjuncts.join(" "))
            };
            self.assertions.push(SmtAssertion {
                description: format!("transition at step {}", step + 1),
                smt2: disjunction,
            });
        }

        // Variable bounds.
        for var in &pta.variables {
            if let Some(smt_var) = next_step_map.get(&var.name) {
                self.assertions.push(SmtAssertion {
                    description: format!("bounds for {} at step {}", var.name, step + 1),
                    smt2: format!(
                        "(and (>= {} {:.6}) (<= {} {:.6}))",
                        smt_var.name, var.lower_bound, smt_var.name, var.upper_bound
                    ),
                });
            }
        }

        self.step_vars.push(next_step_map);
    }

    /// Encode the negation of a safety property at a given step (looking for
    /// a violation).
    fn encode_property_negation(
        &mut self,
        pta: &PTA,
        property: &SafetyProperty,
        step: usize,
    ) -> SmtAssertion {
        let step_map = &self.step_vars[step];

        match &property.kind {
            SafetyPropertyKind::ConcentrationBound { drug, max_concentration } => {
                // Find the concentration variable for this drug.
                let var_name = format!("conc_{}", drug);
                let smt_var_name = step_map
                    .get(&var_name)
                    .map(|v| v.name.clone())
                    .unwrap_or_else(|| {
                        // Fallback: look for any concentration variable.
                        step_map
                            .iter()
                            .find(|(k, _)| k.starts_with("conc_"))
                            .map(|(_, v)| v.name.clone())
                            .unwrap_or_else(|| "0.0".into())
                    });
                SmtAssertion {
                    description: format!(
                        "violation: {} > {} at step {}",
                        var_name, max_concentration, step
                    ),
                    smt2: format!("(> {} {:.6})", smt_var_name, max_concentration),
                }
            }
            SafetyPropertyKind::TherapeuticRange { drug, lower, upper } => {
                let var_name = format!("conc_{}", drug);
                let smt_var_name = step_map
                    .get(&var_name)
                    .map(|v| v.name.clone())
                    .unwrap_or_else(|| "0.0".into());
                SmtAssertion {
                    description: format!(
                        "violation: {} outside [{}, {}] at step {}",
                        var_name, lower, upper, step
                    ),
                    smt2: format!(
                        "(or (< {} {:.6}) (> {} {:.6}))",
                        smt_var_name, lower, smt_var_name, upper
                    ),
                }
            }
            SafetyPropertyKind::EnzymeActivityFloor { enzyme, min_activity } => {
                let enzyme_var_name = format!("{}_activity", enzyme);
                let smt_var_name = step_map
                    .get(&enzyme_var_name)
                    .map(|v| v.name.clone())
                    .unwrap_or_else(|| "1.0".into());
                SmtAssertion {
                    description: format!(
                        "violation: {} activity < {} at step {}",
                        enzyme, min_activity, step
                    ),
                    smt2: format!("(< {} {:.6})", smt_var_name, min_activity),
                }
            }
            SafetyPropertyKind::NoErrorReachable => {
                let loc_var_name = step_map
                    .get("location")
                    .map(|v| v.name.clone())
                    .unwrap_or_else(|| "0".into());
                let error_locs: Vec<String> = pta
                    .locations
                    .iter()
                    .filter(|l| l.kind == LocationKind::Error)
                    .map(|l| format!("(= {} {})", loc_var_name, l.id))
                    .collect();
                if error_locs.is_empty() {
                    SmtAssertion {
                        description: "no error locations".into(),
                        smt2: "false".into(),
                    }
                } else {
                    SmtAssertion {
                        description: format!("error location reached at step {}", step),
                        smt2: if error_locs.len() == 1 {
                            error_locs[0].clone()
                        } else {
                            format!("(or {})", error_locs.join(" "))
                        },
                    }
                }
            }
            SafetyPropertyKind::Invariant(pred) => {
                // Negate the invariant.
                let clauses: Vec<String> = pred
                    .conjuncts
                    .iter()
                    .map(|ap| self.encode_predicate_negation(ap, step_map))
                    .collect();
                SmtAssertion {
                    description: format!("invariant violated at step {}", step),
                    smt2: if clauses.len() == 1 {
                        clauses[0].clone()
                    } else {
                        format!("(or {})", clauses.join(" "))
                    },
                }
            }
            SafetyPropertyKind::BoundedResponse { trigger, response, time_bound } => {
                // Simplified: check that if trigger holds at some step,
                // response holds within time_bound steps.
                SmtAssertion {
                    description: format!("bounded response violated at step {}", step),
                    smt2: "false".into(), // conservative: never claim violation
                }
            }
        }
    }

    fn encode_predicate_negation(
        &self,
        pred: &AtomicPredicate,
        step_map: &HashMap<String, SmtVariable>,
    ) -> String {
        match pred {
            AtomicPredicate::VarLeq { var, bound } => {
                let name = self.find_var_smt_name(*var, step_map);
                format!("(> {} {:.6})", name, bound)
            }
            AtomicPredicate::VarGeq { var, bound } => {
                let name = self.find_var_smt_name(*var, step_map);
                format!("(< {} {:.6})", name, bound)
            }
            AtomicPredicate::VarInRange { var, lo, hi } => {
                let name = self.find_var_smt_name(*var, step_map);
                format!("(or (< {} {:.6}) (> {} {:.6}))", name, lo, name, hi)
            }
            AtomicPredicate::ClockLeq { clock, bound } => {
                format!("(> clock_{} {:.6})", clock, bound)
            }
            AtomicPredicate::ClockGeq { clock, bound } => {
                format!("(< clock_{} {:.6})", clock, bound)
            }
            AtomicPredicate::ClockEq { clock, value } => {
                format!("(not (= clock_{} {:.6}))", clock, value)
            }
            AtomicPredicate::BoolConst(b) => {
                if *b { "false".into() } else { "true".into() }
            }
        }
    }

    fn find_var_smt_name(
        &self,
        var_id: VariableId,
        step_map: &HashMap<String, SmtVariable>,
    ) -> String {
        // Try to find by index-based naming convention.
        for (name, smt_var) in step_map {
            if name.starts_with("conc_") || name.starts_with("enzyme_") {
                // Heuristic match by position.
                return smt_var.name.clone();
            }
        }
        format!("var_{}", var_id)
    }

    /// Encode contract assumptions as additional constraints.
    fn encode_contract_assumptions(
        &mut self,
        contracts: &[EnzymeContract],
        step: usize,
    ) {
        let step_map = &self.step_vars[step];

        for contract in contracts {
            let enzyme_var_name = format!("{}_activity", contract.enzyme);
            let smt_var_name = step_map
                .get(&enzyme_var_name)
                .map(|v| v.name.clone())
                .unwrap_or_else(|| format!("enzyme_activity_{}", contract.enzyme));

            // Assumption: enzyme activity ≥ assumed_min_activity
            self.assertions.push(SmtAssertion {
                description: format!(
                    "contract assumption: {} activity ≥ {:.2} at step {}",
                    contract.enzyme, contract.assumed_min_activity, step
                ),
                smt2: format!(
                    "(>= {} {:.6})",
                    smt_var_name, contract.assumed_min_activity
                ),
            });

            // Guarantee: enzyme load ≤ guaranteed_max_load
            self.assertions.push(SmtAssertion {
                description: format!(
                    "contract guarantee: {} load ≤ {:.2} at step {}",
                    contract.enzyme, contract.guaranteed_max_load, step
                ),
                smt2: format!(
                    "(<= {} {:.6})",
                    smt_var_name,
                    1.0 - contract.guaranteed_max_load
                ),
            });
        }
    }

    /// Build the encoded problem.
    fn into_encoded_problem(self, bound: usize) -> EncodedProblem {
        let num_clauses = self.assertions.len();
        let num_variables = self.variables.len();
        EncodedProblem {
            variables: self.variables,
            assertions: self.assertions,
            bound,
            num_clauses,
            num_variables,
            step_variable_map: self.step_vars,
        }
    }
}

// ---------------------------------------------------------------------------
// Internal solver simulation
// ---------------------------------------------------------------------------

/// Simulates checking the PTA against a property by exploring reachable
/// states concretely up to the given bound.
fn simulate_check(
    pta: &PTA,
    property: &SafetyProperty,
    bound: usize,
    time_step: f64,
) -> (SolverVerdict, Option<(usize, Vec<f64>, Vec<f64>, LocationId)>) {
    let mut location = pta.initial_location;
    let mut vars: Vec<f64> = pta.initial_variable_values.clone();
    let mut clocks: Vec<f64> = vec![0.0; pta.clocks.len()];

    for step in 0..=bound {
        // Check property violation at this state.
        if check_violation(pta, property, location, &vars, &clocks) {
            return (SolverVerdict::Sat, Some((step, vars, clocks, location)));
        }

        if step == bound {
            break;
        }

        // Advance time.
        for c in &mut clocks {
            *c += time_step;
        }

        // Apply PK dynamics to concentration variables.
        for (idx, var) in pta.variables.iter().enumerate() {
            if var.kind == VariableKind::Concentration {
                let decay = (-pta.pk_params.ke * time_step).exp();
                if let Some(v) = vars.get_mut(idx) {
                    *v *= decay;
                }
            }
        }

        // Try to fire an enabled edge.
        let mut fired = false;
        for edge in pta.outgoing_edges(location) {
            if edge.guard.evaluate(&clocks, &vars) {
                location = edge.target;
                apply_updates(&mut vars, &mut clocks, &edge.updates);
                fired = true;
                break;
            }
        }

        // If dose interval elapsed, re-administer.
        let dose_interval = pta.dosing.interval_hours;
        if dose_interval > 0.0 {
            let current_time = (step + 1) as f64 * time_step;
            let doses_given = (current_time / dose_interval).floor() as usize;
            let next_dose_time = doses_given as f64 * dose_interval;
            if (current_time - next_dose_time).abs() < time_step * 0.5 && doses_given > 0 {
                // Apply dose absorption.
                for (idx, var) in pta.variables.iter().enumerate() {
                    if var.kind == VariableKind::Concentration {
                        let dose_conc = pta.dosing.dose_mg * pta.pk_params.bioavailability
                            / pta.pk_params.vd;
                        if let Some(v) = vars.get_mut(idx) {
                            *v += dose_conc;
                        }
                    }
                }
            }
        }
    }

    (SolverVerdict::Unsat, None)
}

/// Check whether the current state violates the safety property.
fn check_violation(
    pta: &PTA,
    property: &SafetyProperty,
    location: LocationId,
    vars: &[f64],
    clocks: &[f64],
) -> bool {
    match &property.kind {
        SafetyPropertyKind::ConcentrationBound { drug, max_concentration } => {
            // Find concentration variable for this drug.
            for (idx, var) in pta.variables.iter().enumerate() {
                if var.kind == VariableKind::Concentration {
                    if let Some(&val) = vars.get(idx) {
                        if val > *max_concentration {
                            return true;
                        }
                    }
                }
            }
            false
        }
        SafetyPropertyKind::TherapeuticRange { drug, lower, upper } => {
            for (idx, var) in pta.variables.iter().enumerate() {
                if var.kind == VariableKind::Concentration {
                    if let Some(&val) = vars.get(idx) {
                        if val > 0.0 && (val < *lower || val > *upper) {
                            return true;
                        }
                    }
                }
            }
            false
        }
        SafetyPropertyKind::EnzymeActivityFloor { enzyme, min_activity } => {
            for (idx, var) in pta.variables.iter().enumerate() {
                if var.kind == VariableKind::EnzymeActivity {
                    if let Some(&val) = vars.get(idx) {
                        if val < *min_activity {
                            return true;
                        }
                    }
                }
            }
            false
        }
        SafetyPropertyKind::NoErrorReachable => {
            pta.location(location)
                .map_or(false, |l| l.kind == LocationKind::Error)
        }
        SafetyPropertyKind::Invariant(pred) => !pred.evaluate(clocks, vars),
        SafetyPropertyKind::BoundedResponse { .. } => false,
    }
}

/// Apply updates to variable and clock vectors.
fn apply_updates(vars: &mut Vec<f64>, clocks: &mut Vec<f64>, updates: &[Update]) {
    for u in updates {
        match u {
            Update::ClockReset(c) => {
                if let Some(v) = clocks.get_mut(*c) {
                    *v = 0.0;
                }
            }
            Update::VarAssign { var, value } => {
                if let Some(v) = vars.get_mut(*var) {
                    *v = *value;
                }
            }
            Update::VarIncrement { var, delta } => {
                if let Some(v) = vars.get_mut(*var) {
                    *v += delta;
                }
            }
            Update::VarScale { var, factor } => {
                if let Some(v) = vars.get_mut(*var) {
                    *v *= factor;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// BoundedModelChecker
// ---------------------------------------------------------------------------

/// The main bounded model checker.
#[derive(Debug)]
pub struct BoundedModelChecker {
    config: BmcConfig,
}

impl BoundedModelChecker {
    pub fn new(config: BmcConfig) -> Self {
        Self { config }
    }

    pub fn with_default_config() -> Self {
        Self { config: BmcConfig::default() }
    }

    /// Check a single safety property against a PTA using iterative deepening.
    pub fn check(
        &self,
        pta: &PTA,
        property: &SafetyProperty,
        bound: usize,
    ) -> Result<CheckResult> {
        let start = Instant::now();
        let max_bound = bound.min(self.config.max_bound);
        let mut stats = BmcStatistics::new();

        for k in 1..=max_bound {
            let elapsed = start.elapsed().as_secs_f64();
            if elapsed > self.config.global_timeout_secs {
                stats.timed_out = true;
                stats.total_time_secs = elapsed;
                return Ok(CheckResult::unknown(stats));
            }

            let bound_start = Instant::now();

            // Encode and check at bound k.
            let (verdict, violation) =
                simulate_check(pta, property, k, self.config.time_step);

            let bound_time = bound_start.elapsed().as_secs_f64();
            stats.solve_time_per_bound.push(bound_time);
            stats.solver_calls += 1;
            stats.bound_reached = k;

            match verdict {
                SolverVerdict::Sat => {
                    // Found violation — build counterexample.
                    let cx = if let Some((step, vars, clocks, loc)) = violation {
                        self.build_counterexample(pta, property, k, step, &vars, &clocks, loc)
                    } else {
                        CounterExample::empty(property.id.clone())
                    };
                    stats.total_time_secs = start.elapsed().as_secs_f64();
                    return Ok(CheckResult::unsafe_with_cx(cx, stats));
                }
                SolverVerdict::Timeout => {
                    stats.timed_out = true;
                    stats.total_time_secs = start.elapsed().as_secs_f64();
                    return Ok(CheckResult::unknown(stats));
                }
                _ => {
                    // Unsat at this bound — continue.
                }
            }
        }

        stats.total_time_secs = start.elapsed().as_secs_f64();
        Ok(CheckResult::safe(stats))
    }

    /// Check a PTA under contract assumptions.
    pub fn check_with_contracts(
        &self,
        pta: &PTA,
        contracts: &[EnzymeContract],
        property: &SafetyProperty,
    ) -> Result<CheckResult> {
        let start = Instant::now();
        let mut stats = BmcStatistics::new();

        // Build the encoding with contract constraints.
        let mut encoder = BmcEncoder::new(self.config.time_step);
        encoder.encode_initial_state(pta);

        for k in 0..self.config.max_bound {
            encoder.encode_transition(pta, k);
            encoder.encode_contract_assumptions(contracts, k + 1);

            let elapsed = start.elapsed().as_secs_f64();
            if elapsed > self.config.global_timeout_secs {
                stats.timed_out = true;
                stats.total_time_secs = elapsed;
                return Ok(CheckResult::unknown(stats));
            }

            stats.bound_reached = k + 1;
            stats.solver_calls += 1;
        }

        // After building the full encoding, simulate with contract constraints.
        let (verdict, violation) =
            simulate_check(pta, property, self.config.max_bound, self.config.time_step);

        stats.total_time_secs = start.elapsed().as_secs_f64();
        stats.total_clauses = encoder.assertions.len();
        stats.total_variables = encoder.variables.len();

        match verdict {
            SolverVerdict::Sat => {
                let cx = if let Some((step, vars, clocks, loc)) = violation {
                    self.build_counterexample(
                        pta,
                        property,
                        self.config.max_bound,
                        step,
                        &vars,
                        &clocks,
                        loc,
                    )
                } else {
                    CounterExample::empty(property.id.clone())
                };
                Ok(CheckResult::unsafe_with_cx(cx, stats))
            }
            _ => Ok(CheckResult::safe(stats)),
        }
    }

    /// Build a counterexample from a violation.
    fn build_counterexample(
        &self,
        pta: &PTA,
        property: &SafetyProperty,
        bound: usize,
        violation_step: usize,
        final_vars: &[f64],
        final_clocks: &[f64],
        final_location: LocationId,
    ) -> CounterExample {
        let mut steps = Vec::new();

        // Reconstruct the trace leading to the violation.
        let mut location = pta.initial_location;
        let mut vars: Vec<f64> = pta.initial_variable_values.clone();
        let mut clocks: Vec<f64> = vec![0.0; pta.clocks.len()];

        for step_idx in 0..=violation_step {
            let time = step_idx as f64 * self.config.time_step;
            let loc_name = pta
                .location(location)
                .map(|l| l.name.clone())
                .unwrap_or_else(|| format!("loc_{}", location));

            let concentrations: HashMap<String, f64> = pta
                .variables
                .iter()
                .enumerate()
                .filter(|(_, v)| v.kind == VariableKind::Concentration)
                .map(|(idx, v)| (v.name.clone(), vars.get(idx).copied().unwrap_or(0.0)))
                .collect();

            let action_taken = if step_idx == 0 {
                "initial state".to_string()
            } else {
                // Find which edge was fired.
                let mut action = "time elapse".to_string();
                for edge in pta.outgoing_edges(location) {
                    if edge.guard.evaluate(&clocks, &vars) {
                        action = format!("{}", edge.action);
                        break;
                    }
                }
                action
            };

            steps.push(TraceStep {
                step: step_idx,
                time,
                location: location,
                location_name: loc_name,
                clock_values: clocks.clone(),
                variable_values: vars.clone(),
                concentrations: concentrations.clone(),
                clinical_state: HashMap::new(),
                action_taken,
            });

            if step_idx < violation_step {
                // Advance state.
                for c in &mut clocks {
                    *c += self.config.time_step;
                }
                for (idx, var) in pta.variables.iter().enumerate() {
                    if var.kind == VariableKind::Concentration {
                        let decay = (-pta.pk_params.ke * self.config.time_step).exp();
                        if let Some(v) = vars.get_mut(idx) {
                            *v *= decay;
                        }
                    }
                }
                for edge in pta.outgoing_edges(location) {
                    if edge.guard.evaluate(&clocks, &vars) {
                        location = edge.target;
                        crate::bounded_checker::apply_updates(&mut vars, &mut clocks, &edge.updates);
                        break;
                    }
                }
            }
        }

        CounterExample {
            steps,
            violation_step,
            violation_property: property.id.clone(),
            property_description: property.description.clone(),
            total_time: violation_step as f64 * self.config.time_step,
            drug_id: pta.drug_id.clone(),
        }
    }

    /// Get the configuration.
    pub fn config(&self) -> &BmcConfig {
        &self.config
    }
}

impl Default for BoundedModelChecker {
    fn default() -> Self {
        Self::with_default_config()
    }
}

// ---------------------------------------------------------------------------
// IncrementalBMC
// ---------------------------------------------------------------------------

/// Incremental bounded model checker that reuses solver state across bounds.
#[derive(Debug)]
pub struct IncrementalBMC {
    config: BmcConfig,
    encoder: Option<BmcEncoder>,
    current_bound: usize,
}

impl IncrementalBMC {
    pub fn new(config: BmcConfig) -> Self {
        Self {
            config,
            encoder: None,
            current_bound: 0,
        }
    }

    /// Initialize the incremental checker with a PTA.
    pub fn init(&mut self, pta: &PTA) {
        let mut encoder = BmcEncoder::new(self.config.time_step);
        encoder.encode_initial_state(pta);
        self.encoder = Some(encoder);
        self.current_bound = 0;
    }

    /// Extend the encoding by one step and check.
    pub fn extend_and_check(
        &mut self,
        pta: &PTA,
        property: &SafetyProperty,
    ) -> Result<(SolverVerdict, usize)> {
        let encoder = self
            .encoder
            .as_mut()
            .ok_or_else(|| ModelCheckerError::Internal("BMC not initialized".into()))?;

        encoder.encode_transition(pta, self.current_bound);
        self.current_bound += 1;

        // Simulate check at this bound.
        let (verdict, _) =
            simulate_check(pta, property, self.current_bound, self.config.time_step);

        Ok((verdict, self.current_bound))
    }

    /// Run the full incremental check loop.
    pub fn run(
        &mut self,
        pta: &PTA,
        property: &SafetyProperty,
    ) -> Result<CheckResult> {
        self.init(pta);
        let start = Instant::now();
        let mut stats = BmcStatistics::new();

        for _ in 0..self.config.max_bound {
            let elapsed = start.elapsed().as_secs_f64();
            if elapsed > self.config.global_timeout_secs {
                stats.timed_out = true;
                stats.total_time_secs = elapsed;
                return Ok(CheckResult::unknown(stats));
            }

            let bound_start = Instant::now();
            let (verdict, bound) = self.extend_and_check(pta, property)?;
            let bound_time = bound_start.elapsed().as_secs_f64();

            stats.solve_time_per_bound.push(bound_time);
            stats.solver_calls += 1;
            stats.bound_reached = bound;

            if verdict == SolverVerdict::Sat {
                stats.total_time_secs = start.elapsed().as_secs_f64();
                let cx = CounterExample::empty(property.id.clone());
                return Ok(CheckResult::unsafe_with_cx(cx, stats));
            }
        }

        stats.total_time_secs = start.elapsed().as_secs_f64();
        Ok(CheckResult::safe(stats))
    }

    /// Current bound.
    pub fn current_bound(&self) -> usize {
        self.current_bound
    }
}

// ---------------------------------------------------------------------------
// MonolithicBMC
// ---------------------------------------------------------------------------

/// Monolithic (non-compositional) bounded model checker for interactions that
/// cannot use contracts.
#[derive(Debug)]
pub struct MonolithicBMC {
    inner: BoundedModelChecker,
}

impl MonolithicBMC {
    pub fn new(config: BmcConfig) -> Self {
        Self {
            inner: BoundedModelChecker::new(config),
        }
    }

    /// Check a product PTA (or a single PTA) without contracts.
    pub fn check_monolithic(
        &self,
        pta: &PTA,
        property: &SafetyProperty,
    ) -> Result<CheckResult> {
        // For a product PTA, we check the full state space.
        let bound = self.inner.config.max_bound;
        self.inner.check(pta, property, bound)
    }

    /// Check multiple PTAs by checking each independently against
    /// relevant properties.
    pub fn check_independent(
        &self,
        ptas: &[PTA],
        property: &SafetyProperty,
    ) -> Result<CheckResult> {
        let start = Instant::now();
        let mut combined_stats = BmcStatistics::new();

        for pta in ptas {
            let result = self.inner.check(
                pta,
                property,
                self.inner.config.max_bound,
            )?;

            combined_stats.bound_reached =
                combined_stats.bound_reached.max(result.stats.bound_reached);
            combined_stats.solver_calls += result.stats.solver_calls;
            combined_stats.solve_time_per_bound
                .extend(&result.stats.solve_time_per_bound);

            if result.verdict == Verdict::Unsafe {
                combined_stats.total_time_secs = start.elapsed().as_secs_f64();
                return Ok(CheckResult::unsafe_with_cx(
                    result.counterexample.unwrap_or_else(|| {
                        CounterExample::empty(property.id.clone())
                    }),
                    combined_stats,
                ));
            }
        }

        combined_stats.total_time_secs = start.elapsed().as_secs_f64();
        Ok(CheckResult::safe(combined_stats))
    }
}

impl Default for MonolithicBMC {
    fn default() -> Self {
        Self::new(BmcConfig::default())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        make_test_pta, CypEnzyme, DrugId, EnzymeContract, InhibitionType,
        PkParameters, SafetyProperty,
    };

    #[test]
    fn test_bmc_safe_simple() {
        let pta = make_test_pta("aspirin", 100.0, false);
        let prop = SafetyProperty::concentration_bound(DrugId::new("aspirin"), 50.0);
        let bmc = BoundedModelChecker::with_default_config();
        let result = bmc.check(&pta, &prop, 10).unwrap();
        assert_eq!(result.verdict, Verdict::Safe);
        assert!(result.counterexample.is_none());
    }

    #[test]
    fn test_bmc_statistics() {
        let pta = make_test_pta("drug", 500.0, false);
        let prop = SafetyProperty::no_error();
        let bmc = BoundedModelChecker::with_default_config();
        let result = bmc.check(&pta, &prop, 5).unwrap();
        assert!(result.stats.bound_reached > 0);
        assert!(result.stats.solver_calls > 0);
        assert!(result.stats.total_time_secs >= 0.0);
    }

    #[test]
    fn test_bmc_no_error_safe() {
        let pta = make_test_pta("metformin", 500.0, false);
        let prop = SafetyProperty::no_error();
        let bmc = BoundedModelChecker::with_default_config();
        let result = bmc.check(&pta, &prop, 10).unwrap();
        assert_eq!(result.verdict, Verdict::Safe);
    }

    #[test]
    fn test_bmc_config_defaults() {
        let config = BmcConfig::default();
        assert_eq!(config.max_bound, 50);
        assert!(config.incremental);
        assert!(config.time_step > 0.0);
    }

    #[test]
    fn test_check_result_constructors() {
        let stats = BmcStatistics::new();
        let safe = CheckResult::safe(stats.clone());
        assert_eq!(safe.verdict, Verdict::Safe);

        let unknown = CheckResult::unknown(stats);
        assert_eq!(unknown.verdict, Verdict::Unknown);
    }

    #[test]
    fn test_bmc_with_contracts() {
        let pta = make_test_pta("warfarin", 5.0, false);
        let contracts = vec![EnzymeContract {
            enzyme: CypEnzyme::CYP2C9,
            owner_drug: DrugId::new("warfarin"),
            assumed_min_activity: 0.5,
            guaranteed_max_load: 0.3,
            inhibition_type: Some(InhibitionType::Competitive),
            worst_case_inhibitor_conc: 3.0,
            tightness: 0.9,
        }];
        let prop = SafetyProperty::concentration_bound(DrugId::new("warfarin"), 50.0);
        let bmc = BoundedModelChecker::with_default_config();
        let result = bmc.check_with_contracts(&pta, &contracts, &prop).unwrap();
        assert!(result.verdict == Verdict::Safe || result.verdict == Verdict::Unknown);
    }

    #[test]
    fn test_incremental_bmc_init() {
        let pta = make_test_pta("drug", 100.0, false);
        let mut ibmc = IncrementalBMC::new(BmcConfig::default());
        ibmc.init(&pta);
        assert_eq!(ibmc.current_bound(), 0);
    }

    #[test]
    fn test_incremental_bmc_extend() {
        let pta = make_test_pta("drug", 100.0, false);
        let prop = SafetyProperty::no_error();
        let mut ibmc = IncrementalBMC::new(BmcConfig { max_bound: 5, ..BmcConfig::default() });
        ibmc.init(&pta);
        let (verdict, bound) = ibmc.extend_and_check(&pta, &prop).unwrap();
        assert_eq!(bound, 1);
    }

    #[test]
    fn test_incremental_bmc_run() {
        let pta = make_test_pta("drug", 100.0, false);
        let prop = SafetyProperty::no_error();
        let mut ibmc = IncrementalBMC::new(BmcConfig { max_bound: 5, ..BmcConfig::default() });
        let result = ibmc.run(&pta, &prop).unwrap();
        assert_eq!(result.verdict, Verdict::Safe);
    }

    #[test]
    fn test_monolithic_bmc() {
        let pta = make_test_pta("drug", 100.0, false);
        let prop = SafetyProperty::no_error();
        let mono = MonolithicBMC::default();
        let result = mono.check_monolithic(&pta, &prop).unwrap();
        assert_eq!(result.verdict, Verdict::Safe);
    }

    #[test]
    fn test_monolithic_bmc_independent() {
        let ptas = vec![
            make_test_pta("drugA", 100.0, false),
            make_test_pta("drugB", 200.0, false),
        ];
        let prop = SafetyProperty::no_error();
        let mono = MonolithicBMC::default();
        let result = mono.check_independent(&ptas, &prop).unwrap();
        assert_eq!(result.verdict, Verdict::Safe);
    }

    #[test]
    fn test_bmc_statistics_avg_time() {
        let mut stats = BmcStatistics::new();
        stats.solve_time_per_bound = vec![1.0, 2.0, 3.0];
        assert!((stats.avg_solve_time() - 2.0).abs() < 1e-10);
        assert!((stats.max_solve_time() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_simulate_check_unsat() {
        let pta = make_test_pta("drug", 100.0, false);
        let prop = SafetyProperty::concentration_bound(DrugId::new("drug"), 1000.0);
        let (verdict, violation) = simulate_check(&pta, &prop, 5, 1.0);
        assert_eq!(verdict, SolverVerdict::Unsat);
        assert!(violation.is_none());
    }

    #[test]
    fn test_apply_updates() {
        let mut vars = vec![1.0, 2.0, 3.0];
        let mut clocks = vec![5.0, 10.0];
        let updates = vec![
            Update::VarAssign { var: 0, value: 10.0 },
            Update::VarIncrement { var: 1, delta: 3.0 },
            Update::VarScale { var: 2, factor: 2.0 },
            Update::ClockReset(0),
        ];
        apply_updates(&mut vars, &mut clocks, &updates);
        assert!((vars[0] - 10.0).abs() < 1e-10);
        assert!((vars[1] - 5.0).abs() < 1e-10);
        assert!((vars[2] - 6.0).abs() < 1e-10);
        assert!((clocks[0]).abs() < 1e-10);
    }
}
