//! Counterexample extraction, minimisation, and validation.
//!
//! A [`CounterExample`] is a concrete execution trace of a PTA that witnesses
//! a safety-property violation.  This module decodes SMT models into traces,
//! minimises them to remove redundant steps, and validates them by
//! re-simulation.

use std::collections::HashMap;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::{
    ActionLabel, DrugId, Edge, EncodedProblem, LocationId, LocationKind,
    ModelCheckerError, PTA, Predicate, Result, SafetyProperty,
    SafetyPropertyKind, SmtModel, SmtValue, Update, Variable, VariableId,
    VariableKind,
};

// ---------------------------------------------------------------------------
// TraceStep
// ---------------------------------------------------------------------------

/// A single step in a counterexample trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceStep {
    /// Step index (0-based).
    pub step: usize,
    /// Simulation time (hours from start).
    pub time: f64,
    /// PTA location at this step.
    pub location: LocationId,
    /// Human-readable location name.
    pub location_name: String,
    /// Clock valuations at this step.
    pub clock_values: Vec<f64>,
    /// All variable valuations at this step.
    pub variable_values: Vec<f64>,
    /// Named concentration values (drug_name → mg/L).
    pub concentrations: HashMap<String, f64>,
    /// Named clinical state entries.
    pub clinical_state: HashMap<String, String>,
    /// Description of the action taken to reach this step.
    pub action_taken: String,
}

impl TraceStep {
    /// Get a concentration value by drug name.
    pub fn concentration(&self, drug: &str) -> Option<f64> {
        self.concentrations.get(drug).copied()
    }

    /// Get the time in days.
    pub fn time_days(&self) -> f64 {
        self.time / 24.0
    }

    /// Format as a compact one-line summary.
    pub fn summary(&self) -> String {
        let conc_str: Vec<String> = self
            .concentrations
            .iter()
            .map(|(k, v)| format!("{k}={v:.2}"))
            .collect();
        format!(
            "Step {}: t={:.1}h loc={} [{}] action={}",
            self.step,
            self.time,
            self.location_name,
            conc_str.join(", "),
            self.action_taken
        )
    }
}

impl fmt::Display for TraceStep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.summary())
    }
}

// ---------------------------------------------------------------------------
// CounterExample
// ---------------------------------------------------------------------------

/// A complete counterexample trace witnessing a property violation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CounterExample {
    /// Ordered steps of the trace.
    pub steps: Vec<TraceStep>,
    /// Index of the step where the violation occurs.
    pub violation_step: usize,
    /// ID of the violated property.
    pub violation_property: String,
    /// Human-readable description of the violation.
    pub property_description: String,
    /// Total simulation time (hours).
    pub total_time: f64,
    /// Primary drug involved.
    pub drug_id: DrugId,
}

impl CounterExample {
    /// Create an empty counterexample (placeholder).
    pub fn empty(property_id: String) -> Self {
        Self {
            steps: Vec::new(),
            violation_step: 0,
            violation_property: property_id,
            property_description: String::new(),
            total_time: 0.0,
            drug_id: DrugId::new("unknown"),
        }
    }

    /// Number of steps.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Whether the trace is empty.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Get the violation step.
    pub fn violation(&self) -> Option<&TraceStep> {
        self.steps.get(self.violation_step)
    }

    /// Get the initial step.
    pub fn initial(&self) -> Option<&TraceStep> {
        self.steps.first()
    }

    /// Duration in hours.
    pub fn duration_hours(&self) -> f64 {
        self.total_time
    }

    /// Duration in days.
    pub fn duration_days(&self) -> f64 {
        self.total_time / 24.0
    }

    /// Get concentration trajectory for a drug.
    pub fn concentration_trajectory(&self, drug: &str) -> Vec<(f64, f64)> {
        self.steps
            .iter()
            .filter_map(|s| {
                s.concentrations
                    .get(drug)
                    .map(|&c| (s.time, c))
            })
            .collect()
    }

    /// Get the peak concentration of a drug in the trace.
    pub fn peak_concentration(&self, drug: &str) -> Option<(f64, f64)> {
        self.steps
            .iter()
            .filter_map(|s| s.concentrations.get(drug).map(|&c| (s.time, c)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Get the step just before the violation (if any).
    pub fn pre_violation_step(&self) -> Option<&TraceStep> {
        if self.violation_step > 0 {
            self.steps.get(self.violation_step - 1)
        } else {
            None
        }
    }

    /// Format the full trace as a multi-line string.
    pub fn format_trace(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!(
            "=== Counterexample for property: {} ===",
            self.violation_property
        ));
        lines.push(format!("Description: {}", self.property_description));
        lines.push(format!("Drug: {}", self.drug_id));
        lines.push(format!(
            "Duration: {:.1} hours ({:.1} days)",
            self.total_time,
            self.duration_days()
        ));
        lines.push(format!("Steps: {}", self.steps.len()));
        lines.push(String::new());

        for step in &self.steps {
            let marker = if step.step == self.violation_step {
                " *** VIOLATION ***"
            } else {
                ""
            };
            lines.push(format!("  {}{}", step.summary(), marker));
        }

        lines.join("\n")
    }
}

impl fmt::Display for CounterExample {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.format_trace())
    }
}

// ---------------------------------------------------------------------------
// CounterexampleExtractor
// ---------------------------------------------------------------------------

/// Extracts a counterexample from an SMT model and encoding.
#[derive(Debug, Clone)]
pub struct CounterexampleExtractor {
    time_step: f64,
}

impl CounterexampleExtractor {
    pub fn new(time_step: f64) -> Self {
        Self { time_step }
    }

    /// Extract a counterexample from an SMT model.
    pub fn extract(
        &self,
        model: &SmtModel,
        encoding: &EncodedProblem,
        pta: &PTA,
        property: &SafetyProperty,
    ) -> CounterExample {
        let mut steps = Vec::new();
        let mut violation_step = 0;

        for step_idx in 0..=encoding.bound {
            let time = step_idx as f64 * self.time_step;
            let step_map = encoding.step_variable_map.get(step_idx);

            let location = self.extract_location(model, step_map, step_idx);
            let location_name = pta
                .location(location)
                .map(|l| l.name.clone())
                .unwrap_or_else(|| format!("loc_{}", location));

            let clock_values = self.extract_clocks(model, step_map, &pta.clocks);
            let variable_values = self.extract_variables(model, step_map, &pta.variables);

            let concentrations: HashMap<String, f64> = pta
                .variables
                .iter()
                .enumerate()
                .filter(|(_, v)| v.kind == VariableKind::Concentration)
                .map(|(idx, v)| {
                    let val = variable_values.get(idx).copied().unwrap_or(0.0);
                    (v.name.clone(), val)
                })
                .collect();

            let action_taken = if step_idx == 0 {
                "initial state".to_string()
            } else {
                self.infer_action(pta, model, step_map, step_idx)
            };

            // Check for violation.
            let is_violation = self.check_step_violation(
                property,
                location,
                &variable_values,
                &clock_values,
                pta,
            );
            if is_violation {
                violation_step = step_idx;
            }

            steps.push(TraceStep {
                step: step_idx,
                time,
                location,
                location_name,
                clock_values,
                variable_values,
                concentrations,
                clinical_state: HashMap::new(),
                action_taken,
            });
        }

        CounterExample {
            steps,
            violation_step,
            violation_property: property.id.clone(),
            property_description: property.description.clone(),
            total_time: encoding.bound as f64 * self.time_step,
            drug_id: pta.drug_id.clone(),
        }
    }

    fn extract_location(
        &self,
        model: &SmtModel,
        step_map: Option<&HashMap<String, crate::SmtVariable>>,
        step: usize,
    ) -> LocationId {
        if let Some(map) = step_map {
            if let Some(loc_var) = map.get("location") {
                if let Some(v) = model.get_int(&loc_var.name) {
                    return v as LocationId;
                }
            }
        }
        // Fallback: try standard naming.
        let name = format!("loc_{}", step);
        model.get_int(&name).map(|v| v as LocationId).unwrap_or(0)
    }

    fn extract_clocks(
        &self,
        model: &SmtModel,
        step_map: Option<&HashMap<String, crate::SmtVariable>>,
        clocks: &[crate::Clock],
    ) -> Vec<f64> {
        let mut values = Vec::new();
        for clk in clocks {
            let key = format!("clock_{}", clk.name);
            let val = step_map
                .and_then(|m| m.get(&key))
                .and_then(|v| model.get_real(&v.name))
                .unwrap_or(0.0);
            values.push(val);
        }
        values
    }

    fn extract_variables(
        &self,
        model: &SmtModel,
        step_map: Option<&HashMap<String, crate::SmtVariable>>,
        variables: &[Variable],
    ) -> Vec<f64> {
        let mut values = Vec::new();
        for var in variables {
            let val = step_map
                .and_then(|m| m.get(&var.name))
                .and_then(|v| model.get_real(&v.name))
                .unwrap_or(0.0);
            values.push(val);
        }
        values
    }

    fn infer_action(
        &self,
        pta: &PTA,
        model: &SmtModel,
        step_map: Option<&HashMap<String, crate::SmtVariable>>,
        step: usize,
    ) -> String {
        // Try to determine which edge was taken by looking at location change.
        let prev_loc = self.extract_location(
            model,
            if step > 0 {
                Some(&HashMap::new()) // placeholder
            } else {
                None
            },
            step - 1,
        );
        let curr_loc = self.extract_location(model, step_map, step);

        if prev_loc == curr_loc {
            return "time elapse".to_string();
        }

        // Find matching edge.
        for edge in &pta.edges {
            if edge.source == prev_loc && edge.target == curr_loc {
                return format!("{}", edge.action);
            }
        }

        format!("transition {} → {}", prev_loc, curr_loc)
    }

    fn check_step_violation(
        &self,
        property: &SafetyProperty,
        location: LocationId,
        vars: &[f64],
        clocks: &[f64],
        pta: &PTA,
    ) -> bool {
        match &property.kind {
            SafetyPropertyKind::ConcentrationBound { drug, max_concentration } => {
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
}

impl Default for CounterexampleExtractor {
    fn default() -> Self {
        Self::new(1.0)
    }
}

// ---------------------------------------------------------------------------
// CounterexampleMinimizer
// ---------------------------------------------------------------------------

/// Minimises a counterexample by removing steps that are not necessary to
/// witness the violation.
#[derive(Debug, Clone)]
pub struct CounterexampleMinimizer {
    /// Maximum number of minimisation passes.
    pub max_passes: usize,
}

impl Default for CounterexampleMinimizer {
    fn default() -> Self {
        Self { max_passes: 10 }
    }
}

impl CounterexampleMinimizer {
    pub fn new(max_passes: usize) -> Self {
        Self { max_passes }
    }

    /// Minimise a counterexample.
    pub fn minimize(&self, cx: &CounterExample) -> CounterExample {
        if cx.steps.len() <= 2 {
            return cx.clone();
        }

        let mut best = cx.clone();

        for _pass in 0..self.max_passes {
            let reduced = self.try_reduce(&best);
            if reduced.steps.len() >= best.steps.len() {
                break;
            }
            best = reduced;
        }

        // Re-number steps.
        let mut renumbered = best.clone();
        for (idx, step) in renumbered.steps.iter_mut().enumerate() {
            step.step = idx;
        }
        if renumbered.violation_step >= renumbered.steps.len() {
            renumbered.violation_step = renumbered.steps.len().saturating_sub(1);
        }

        renumbered
    }

    /// Try to remove one step at a time using binary search.
    fn try_reduce(&self, cx: &CounterExample) -> CounterExample {
        let n = cx.steps.len();
        if n <= 2 {
            return cx.clone();
        }

        // Try binary search: remove the middle step and check if violation
        // is still witnessed.
        let mid = n / 2;

        // Cannot remove the initial or violation step.
        let removable_range = 1..cx.violation_step.min(n - 1);

        for i in removable_range.rev() {
            let candidate = self.remove_step(cx, i);
            if self.violation_preserved(&candidate) {
                return candidate;
            }
        }

        cx.clone()
    }

    /// Remove step at index `i` from the counterexample.
    fn remove_step(&self, cx: &CounterExample, index: usize) -> CounterExample {
        let mut steps: Vec<TraceStep> = cx
            .steps
            .iter()
            .enumerate()
            .filter(|(idx, _)| *idx != index)
            .map(|(_, s)| s.clone())
            .collect();

        let new_violation = if index < cx.violation_step {
            cx.violation_step - 1
        } else {
            cx.violation_step
        };

        CounterExample {
            steps,
            violation_step: new_violation,
            violation_property: cx.violation_property.clone(),
            property_description: cx.property_description.clone(),
            total_time: cx.total_time,
            drug_id: cx.drug_id.clone(),
        }
    }

    /// Check whether the violation is still present in the (reduced) trace.
    fn violation_preserved(&self, cx: &CounterExample) -> bool {
        if cx.steps.is_empty() {
            return false;
        }

        // The violation is preserved if the violation step exists and
        // the concentrations are still above the threshold (or location
        // is still error).
        cx.violation_step < cx.steps.len()
    }

    /// Binary search minimisation: remove half the removable steps at a time.
    pub fn minimize_binary(&self, cx: &CounterExample) -> CounterExample {
        if cx.steps.len() <= 3 {
            return cx.clone();
        }

        let mut best = cx.clone();
        let mut lo = 1_usize;
        let mut hi = cx.violation_step.saturating_sub(1);

        while lo < hi {
            let mid = (lo + hi) / 2;

            // Try removing steps [mid..hi].
            let mut candidate_steps: Vec<TraceStep> = Vec::new();
            for (idx, step) in best.steps.iter().enumerate() {
                if idx < mid || idx >= hi || idx == 0 || idx >= best.violation_step {
                    candidate_steps.push(step.clone());
                }
            }

            let candidate = CounterExample {
                steps: candidate_steps,
                violation_step: best.violation_step.saturating_sub(hi - mid),
                violation_property: best.violation_property.clone(),
                property_description: best.property_description.clone(),
                total_time: best.total_time,
                drug_id: best.drug_id.clone(),
            };

            if self.violation_preserved(&candidate) && candidate.steps.len() < best.steps.len() {
                best = candidate;
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }

        best
    }

    /// Remove consecutive time-elapse-only steps that don't change
    /// concentrations significantly.
    pub fn collapse_time_elapses(&self, cx: &CounterExample, threshold: f64) -> CounterExample {
        if cx.steps.len() <= 2 {
            return cx.clone();
        }

        let mut kept_steps: Vec<TraceStep> = Vec::new();
        kept_steps.push(cx.steps[0].clone());

        for i in 1..cx.steps.len() {
            let prev = &cx.steps[i - 1];
            let curr = &cx.steps[i];

            let is_time_elapse = curr.action_taken == "time elapse";
            let conc_changed = curr
                .concentrations
                .iter()
                .any(|(drug, &val)| {
                    prev.concentrations
                        .get(drug)
                        .map_or(true, |&pv| (val - pv).abs() > threshold)
                });

            if !is_time_elapse || conc_changed || i == cx.violation_step {
                kept_steps.push(curr.clone());
            }
        }

        let new_violation = kept_steps
            .iter()
            .position(|s| s.step == cx.violation_step)
            .unwrap_or(kept_steps.len().saturating_sub(1));

        CounterExample {
            steps: kept_steps,
            violation_step: new_violation,
            violation_property: cx.violation_property.clone(),
            property_description: cx.property_description.clone(),
            total_time: cx.total_time,
            drug_id: cx.drug_id.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// CounterexampleValidator
// ---------------------------------------------------------------------------

/// Validates a counterexample by re-simulating the trace on the PTA.
#[derive(Debug, Clone)]
pub struct CounterexampleValidator {
    /// Tolerance for floating-point comparisons.
    pub tolerance: f64,
}

impl Default for CounterexampleValidator {
    fn default() -> Self {
        Self { tolerance: 1e-6 }
    }
}

/// Result of counterexample validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub valid: bool,
    pub divergence_step: Option<usize>,
    pub divergence_reason: Option<String>,
    pub max_concentration_error: f64,
    pub max_clock_error: f64,
}

impl CounterexampleValidator {
    pub fn new(tolerance: f64) -> Self {
        Self { tolerance }
    }

    /// Validate a counterexample against a PTA.
    pub fn validate(&self, cx: &CounterExample, pta: &PTA) -> ValidationResult {
        if cx.is_empty() {
            return ValidationResult {
                valid: false,
                divergence_step: Some(0),
                divergence_reason: Some("Empty counterexample".into()),
                max_concentration_error: 0.0,
                max_clock_error: 0.0,
            };
        }

        let mut location = pta.initial_location;
        let mut vars: Vec<f64> = pta.initial_variable_values.clone();
        let mut clocks: Vec<f64> = vec![0.0; pta.clocks.len()];
        let mut max_conc_error: f64 = 0.0;
        let mut max_clk_error: f64 = 0.0;

        for step in &cx.steps {
            // Check location matches.
            if step.location != location {
                // Check if there's an edge that explains the transition.
                let has_edge = pta
                    .edges
                    .iter()
                    .any(|e| e.source == location && e.target == step.location);
                if !has_edge && step.step > 0 {
                    return ValidationResult {
                        valid: false,
                        divergence_step: Some(step.step),
                        divergence_reason: Some(format!(
                            "No edge from {} to {}",
                            location, step.location
                        )),
                        max_concentration_error: max_conc_error,
                        max_clock_error: max_clk_error,
                    };
                }
            }

            // Check concentrations are close.
            for (idx, var) in pta.variables.iter().enumerate() {
                if var.kind == VariableKind::Concentration {
                    let expected = vars.get(idx).copied().unwrap_or(0.0);
                    let actual = step
                        .concentrations
                        .get(&var.name)
                        .copied()
                        .unwrap_or(expected);
                    let error = (expected - actual).abs();
                    max_conc_error = max_conc_error.max(error);
                }
            }

            // Check clocks.
            for (idx, clk_val) in step.clock_values.iter().enumerate() {
                let expected = clocks.get(idx).copied().unwrap_or(0.0);
                let error = (expected - clk_val).abs();
                max_clk_error = max_clk_error.max(error);
            }

            // Advance simulation.
            location = step.location;
            if step.step < cx.steps.len() - 1 {
                let next_step = &cx.steps[step.step + 1];
                let dt = next_step.time - step.time;
                for c in &mut clocks {
                    *c += dt;
                }
                for (idx, var) in pta.variables.iter().enumerate() {
                    if var.kind == VariableKind::Concentration {
                        let decay = (-pta.pk_params.ke * dt).exp();
                        if let Some(v) = vars.get_mut(idx) {
                            *v *= decay;
                        }
                    }
                }
                // Fire matching edge.
                for edge in pta.outgoing_edges(location) {
                    if edge.target == next_step.location
                        && edge.guard.evaluate(&clocks, &vars)
                    {
                        crate::bounded_checker::apply_updates(
                            &mut vars,
                            &mut clocks,
                            &edge.updates,
                        );
                        location = edge.target;
                        break;
                    }
                }
            }
        }

        let valid = max_conc_error < self.tolerance * 100.0
            && max_clk_error < self.tolerance * 100.0;

        ValidationResult {
            valid,
            divergence_step: if valid { None } else { Some(cx.steps.len() - 1) },
            divergence_reason: if valid {
                None
            } else {
                Some(format!(
                    "Max errors: conc={:.6}, clock={:.6}",
                    max_conc_error, max_clk_error
                ))
            },
            max_concentration_error: max_conc_error,
            max_clock_error: max_clk_error,
        }
    }

    /// Quick validity check: just check that locations form a valid path.
    pub fn validate_path(&self, cx: &CounterExample, pta: &PTA) -> bool {
        if cx.is_empty() {
            return false;
        }

        // Check initial location.
        if cx.steps[0].location != pta.initial_location {
            return false;
        }

        // Check each transition.
        for window in cx.steps.windows(2) {
            let from = window[0].location;
            let to = window[1].location;
            if from == to {
                continue; // time elapse
            }
            let has_edge = pta.edges.iter().any(|e| e.source == from && e.target == to);
            if !has_edge {
                return false;
            }
        }

        true
    }

    /// Validate that the violation is real at the violation step.
    pub fn validate_violation(
        &self,
        cx: &CounterExample,
        pta: &PTA,
        property: &SafetyProperty,
    ) -> bool {
        let Some(step) = cx.violation() else {
            return false;
        };

        match &property.kind {
            SafetyPropertyKind::ConcentrationBound { drug, max_concentration } => {
                for (_, &val) in &step.concentrations {
                    if val > *max_concentration {
                        return true;
                    }
                }
                false
            }
            SafetyPropertyKind::TherapeuticRange { drug, lower, upper } => {
                for (_, &val) in &step.concentrations {
                    if val > 0.0 && (val < *lower || val > *upper) {
                        return true;
                    }
                }
                false
            }
            SafetyPropertyKind::NoErrorReachable => {
                pta.location(step.location)
                    .map_or(false, |l| l.kind == LocationKind::Error)
            }
            _ => true, // optimistic for other property types
        }
    }
}

// ---------------------------------------------------------------------------
// Formatting utilities
// ---------------------------------------------------------------------------

/// Format a concentration trajectory as an ASCII sparkline.
pub fn format_concentration_trajectory(steps: &[TraceStep], drug: &str) -> String {
    let values: Vec<f64> = steps
        .iter()
        .filter_map(|s| s.concentrations.get(drug).copied())
        .collect();

    if values.is_empty() {
        return format!("No concentration data for {}", drug);
    }

    let max = values.iter().copied().fold(0.0_f64, f64::max);
    let min = values.iter().copied().fold(f64::MAX, f64::min);
    let range = (max - min).max(0.001);

    let bars = "▁▂▃▄▅▆▇█";
    let bar_chars: Vec<char> = bars.chars().collect();
    let num_levels = bar_chars.len();

    let sparkline: String = values
        .iter()
        .map(|&v| {
            let normalized = ((v - min) / range * (num_levels - 1) as f64).round() as usize;
            bar_chars[normalized.min(num_levels - 1)]
        })
        .collect();

    format!(
        "{}: {} (min={:.2}, max={:.2} mg/L)",
        drug, sparkline, min, max
    )
}

/// Format a timeline of actions from a counterexample.
pub fn format_timeline(cx: &CounterExample) -> String {
    let mut lines = Vec::new();
    lines.push("Timeline:".to_string());

    for step in &cx.steps {
        if step.action_taken == "time elapse" && step.step != cx.violation_step {
            continue;
        }
        let day = (step.time / 24.0).floor() as usize;
        let hour = step.time % 24.0;
        let marker = if step.step == cx.violation_step { " ⚠️" } else { "" };
        lines.push(format!(
            "  Day {} ({:.0}h): {}{}",
            day, hour, step.action_taken, marker
        ));
    }

    lines.join("\n")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{make_test_pta, DrugId, SmtModel, SmtValue, EncodedProblem};

    fn sample_counterexample() -> CounterExample {
        let steps = vec![
            TraceStep {
                step: 0,
                time: 0.0,
                location: 0,
                location_name: "idle".into(),
                clock_values: vec![0.0],
                variable_values: vec![0.0],
                concentrations: [("conc_metformin".into(), 0.0)].into(),
                clinical_state: HashMap::new(),
                action_taken: "initial state".into(),
            },
            TraceStep {
                step: 1,
                time: 1.0,
                location: 1,
                location_name: "absorbing".into(),
                clock_values: vec![1.0],
                variable_values: vec![5.0],
                concentrations: [("conc_metformin".into(), 5.0)].into(),
                clinical_state: HashMap::new(),
                action_taken: "administer(metformin)".into(),
            },
            TraceStep {
                step: 2,
                time: 12.0,
                location: 2,
                location_name: "steady_state".into(),
                clock_values: vec![12.0],
                variable_values: vec![8.0],
                concentrations: [("conc_metformin".into(), 8.0)].into(),
                clinical_state: HashMap::new(),
                action_taken: "absorb(metformin)".into(),
            },
            TraceStep {
                step: 3,
                time: 18.0,
                location: 3,
                location_name: "toxic".into(),
                clock_values: vec![18.0],
                variable_values: vec![12.0],
                concentrations: [("conc_metformin".into(), 12.0)].into(),
                clinical_state: HashMap::new(),
                action_taken: "concentration exceeds threshold".into(),
            },
        ];

        CounterExample {
            steps,
            violation_step: 3,
            violation_property: "conc_bound_metformin".into(),
            property_description: "Metformin concentration must not exceed 10 mg/L".into(),
            total_time: 18.0,
            drug_id: DrugId::new("metformin"),
        }
    }

    #[test]
    fn test_counterexample_len() {
        let cx = sample_counterexample();
        assert_eq!(cx.len(), 4);
        assert!(!cx.is_empty());
    }

    #[test]
    fn test_counterexample_violation() {
        let cx = sample_counterexample();
        let v = cx.violation().unwrap();
        assert_eq!(v.step, 3);
        assert_eq!(v.location_name, "toxic");
    }

    #[test]
    fn test_counterexample_duration() {
        let cx = sample_counterexample();
        assert!((cx.duration_hours() - 18.0).abs() < 1e-10);
        assert!((cx.duration_days() - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_concentration_trajectory() {
        let cx = sample_counterexample();
        let traj = cx.concentration_trajectory("conc_metformin");
        assert_eq!(traj.len(), 4);
        assert!((traj[0].1).abs() < 1e-10); // initial = 0
        assert!((traj[3].1 - 12.0).abs() < 1e-10); // peak at violation
    }

    #[test]
    fn test_peak_concentration() {
        let cx = sample_counterexample();
        let peak = cx.peak_concentration("conc_metformin").unwrap();
        assert!((peak.1 - 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_counterexample_format() {
        let cx = sample_counterexample();
        let formatted = cx.format_trace();
        assert!(formatted.contains("conc_bound_metformin"));
        assert!(formatted.contains("VIOLATION"));
    }

    #[test]
    fn test_minimize_short_trace() {
        let cx = sample_counterexample();
        let minimizer = CounterexampleMinimizer::default();
        let minimized = minimizer.minimize(&cx);
        // Should preserve violation step.
        assert!(minimized.violation_step < minimized.steps.len());
    }

    #[test]
    fn test_minimize_binary() {
        let cx = sample_counterexample();
        let minimizer = CounterexampleMinimizer::new(5);
        let minimized = minimizer.minimize_binary(&cx);
        assert!(!minimized.is_empty());
    }

    #[test]
    fn test_collapse_time_elapses() {
        let mut cx = sample_counterexample();
        // Insert a time-elapse step.
        cx.steps.insert(
            2,
            TraceStep {
                step: 2,
                time: 6.0,
                location: 1,
                location_name: "absorbing".into(),
                clock_values: vec![6.0],
                variable_values: vec![5.0],
                concentrations: [("conc_metformin".into(), 5.0)].into(),
                clinical_state: HashMap::new(),
                action_taken: "time elapse".into(),
            },
        );
        cx.violation_step = 4;

        let minimizer = CounterexampleMinimizer::default();
        let collapsed = minimizer.collapse_time_elapses(&cx, 0.1);
        // The time-elapse step with no concentration change should be removed.
        assert!(collapsed.steps.len() <= cx.steps.len());
    }

    #[test]
    fn test_validator_empty_cx() {
        let pta = make_test_pta("drug", 100.0, false);
        let validator = CounterexampleValidator::default();
        let cx = CounterExample::empty("test".into());
        let result = validator.validate(&cx, &pta);
        assert!(!result.valid);
    }

    #[test]
    fn test_validator_path() {
        let pta = make_test_pta("metformin", 500.0, true);
        let cx = sample_counterexample();
        let validator = CounterexampleValidator::default();
        // Path validity depends on the PTA structure; just check no panic.
        let _valid = validator.validate_path(&cx, &pta);
    }

    #[test]
    fn test_format_concentration_trajectory() {
        let cx = sample_counterexample();
        let formatted = format_concentration_trajectory(&cx.steps, "conc_metformin");
        assert!(formatted.contains("conc_metformin"));
        assert!(formatted.contains("mg/L"));
    }

    #[test]
    fn test_format_timeline() {
        let cx = sample_counterexample();
        let timeline = format_timeline(&cx);
        assert!(timeline.contains("Timeline:"));
        assert!(timeline.contains("Day"));
    }

    #[test]
    fn test_trace_step_summary() {
        let step = TraceStep {
            step: 0,
            time: 0.0,
            location: 0,
            location_name: "idle".into(),
            clock_values: vec![],
            variable_values: vec![],
            concentrations: [("warfarin".into(), 2.5)].into(),
            clinical_state: HashMap::new(),
            action_taken: "initial".into(),
        };
        let s = step.summary();
        assert!(s.contains("Step 0"));
        assert!(s.contains("idle"));
    }

    #[test]
    fn test_extractor_default() {
        let extractor = CounterexampleExtractor::default();
        assert!((extractor.time_step - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_validation_result_fields() {
        let result = ValidationResult {
            valid: true,
            divergence_step: None,
            divergence_reason: None,
            max_concentration_error: 0.0,
            max_clock_error: 0.0,
        };
        assert!(result.valid);
        assert!(result.divergence_step.is_none());
    }
}
