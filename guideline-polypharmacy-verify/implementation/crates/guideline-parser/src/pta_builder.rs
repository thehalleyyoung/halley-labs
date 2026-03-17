//! Priced Timed Automata (PTA) construction from parsed guideline documents.
//!
//! Converts [`GuidelineDocument`] into a PTA representation that can be
//! consumed by model-checkers or simulation engines.

use crate::format::{
    Branch, ComparisonOp, DecisionPoint, DoseSpec, GuidelineAction, GuidelineDocument,
    GuidelineGuard, MedicationSpec, MonitoringRequirement, SafetyConstraint, TransitionRule,
};
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// Local PTA types (self-contained, no external crate imports)
// ---------------------------------------------------------------------------

/// A location (state) in the PTA.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct Location {
    pub id: String,
    pub name: String,
    pub invariants: Vec<Invariant>,
    pub is_initial: bool,
    pub is_urgent: bool,
    pub is_committed: bool,
    #[serde(default)]
    pub cost_rate: Option<OrderedFloat<f64>>,
    #[serde(default)]
    pub labels: Vec<String>,
}

/// An edge (transition) in the PTA.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Edge {
    pub id: String,
    pub source: String,
    pub target: String,
    pub guard: Guard,
    pub resets: Vec<Reset>,
    pub updates: Vec<Update>,
    #[serde(default)]
    pub sync: Option<String>,
    #[serde(default)]
    pub weight: Option<f64>,
    #[serde(default)]
    pub label: Option<String>,
}

/// A guard (constraint) on an edge.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Guard {
    ClockConstraint {
        clock: String,
        op: ComparisonOp,
        value: f64,
    },
    VariableConstraint {
        variable: String,
        op: ComparisonOp,
        value: f64,
    },
    BooleanVariable {
        variable: String,
        expected: bool,
    },
    RangeConstraint {
        variable: String,
        min: f64,
        max: f64,
    },
    And(Vec<Guard>),
    Or(Vec<Guard>),
    Not(Box<Guard>),
    True,
    False,
}

impl Guard {
    pub fn and(self, other: Self) -> Self {
        match (self, other) {
            (Self::True, g) | (g, Self::True) => g,
            (Self::False, _) | (_, Self::False) => Self::False,
            (Self::And(mut a), Self::And(b)) => {
                a.extend(b);
                Self::And(a)
            }
            (Self::And(mut a), g) => {
                a.push(g);
                Self::And(a)
            }
            (g, Self::And(mut a)) => {
                a.insert(0, g);
                Self::And(a)
            }
            (a, b) => Self::And(vec![a, b]),
        }
    }

    pub fn or(self, other: Self) -> Self {
        match (self, other) {
            (Self::True, _) | (_, Self::True) => Self::True,
            (Self::False, g) | (g, Self::False) => g,
            (Self::Or(mut a), Self::Or(b)) => {
                a.extend(b);
                Self::Or(a)
            }
            (Self::Or(mut a), g) => {
                a.push(g);
                Self::Or(a)
            }
            (g, Self::Or(mut a)) => {
                a.insert(0, g);
                Self::Or(a)
            }
            (a, b) => Self::Or(vec![a, b]),
        }
    }

    pub fn negate(self) -> Self {
        match self {
            Self::True => Self::False,
            Self::False => Self::True,
            Self::Not(inner) => *inner,
            other => Self::Not(Box::new(other)),
        }
    }

    /// Depth of the guard tree.
    pub fn depth(&self) -> usize {
        match self {
            Self::And(gs) | Self::Or(gs) => 1 + gs.iter().map(|g| g.depth()).max().unwrap_or(0),
            Self::Not(g) => 1 + g.depth(),
            _ => 1,
        }
    }

    /// Collect all clock names referenced.
    pub fn clocks(&self) -> HashSet<String> {
        let mut out = HashSet::new();
        self.collect_clocks(&mut out);
        out
    }

    fn collect_clocks(&self, acc: &mut HashSet<String>) {
        match self {
            Self::ClockConstraint { clock, .. } => {
                acc.insert(clock.clone());
            }
            Self::And(gs) | Self::Or(gs) => {
                for g in gs {
                    g.collect_clocks(acc);
                }
            }
            Self::Not(g) => g.collect_clocks(acc),
            _ => {}
        }
    }

    /// Collect all variable names referenced.
    pub fn variables(&self) -> HashSet<String> {
        let mut out = HashSet::new();
        self.collect_variables(&mut out);
        out
    }

    fn collect_variables(&self, acc: &mut HashSet<String>) {
        match self {
            Self::VariableConstraint { variable, .. }
            | Self::BooleanVariable { variable, .. }
            | Self::RangeConstraint { variable, .. } => {
                acc.insert(variable.clone());
            }
            Self::And(gs) | Self::Or(gs) => {
                for g in gs {
                    g.collect_variables(acc);
                }
            }
            Self::Not(g) => g.collect_variables(acc),
            _ => {}
        }
    }
}

/// Clock reset.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Reset {
    pub clock: String,
    pub value: f64,
}

/// Variable update.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Update {
    pub variable: String,
    pub expression: UpdateExpr,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum UpdateExpr {
    Assign(f64),
    Increment(f64),
    SetBool(bool),
    SetString(String),
}

/// An invariant on a location (must hold while residing there).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct Invariant {
    pub clock: String,
    pub op: ComparisonOp,
    pub bound: OrderedFloat<f64>,
}

/// The complete Priced Timed Automaton.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PTA {
    pub name: String,
    pub locations: Vec<Location>,
    pub edges: Vec<Edge>,
    pub clocks: Vec<String>,
    pub variables: Vec<PtaVariable>,
    #[serde(default)]
    pub initial_location: Option<String>,
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PtaVariable {
    pub name: String,
    pub var_type: PtaVarType,
    pub initial_value: f64,
    #[serde(default)]
    pub min: Option<f64>,
    #[serde(default)]
    pub max: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PtaVarType {
    Int,
    Real,
    Bool,
}

impl PTA {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            locations: Vec::new(),
            edges: Vec::new(),
            clocks: Vec::new(),
            variables: Vec::new(),
            initial_location: None,
            metadata: HashMap::new(),
        }
    }

    pub fn find_location(&self, id: &str) -> Option<&Location> {
        self.locations.iter().find(|l| l.id == id)
    }

    pub fn edges_from(&self, loc_id: &str) -> Vec<&Edge> {
        self.edges.iter().filter(|e| e.source == loc_id).collect()
    }

    pub fn edges_to(&self, loc_id: &str) -> Vec<&Edge> {
        self.edges.iter().filter(|e| e.target == loc_id).collect()
    }

    /// All location IDs reachable from the initial location via BFS.
    pub fn reachable_locations(&self) -> HashSet<String> {
        let mut visited = HashSet::new();
        let start = match &self.initial_location {
            Some(s) => s.clone(),
            None => return visited,
        };
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(start.clone());
        visited.insert(start);
        while let Some(current) = queue.pop_front() {
            for edge in self.edges_from(&current) {
                if visited.insert(edge.target.clone()) {
                    queue.push_back(edge.target.clone());
                }
            }
        }
        visited
    }

    /// Number of unreachable locations.
    pub fn unreachable_count(&self) -> usize {
        let reachable = self.reachable_locations();
        self.locations
            .iter()
            .filter(|l| !reachable.contains(&l.id))
            .count()
    }
}

// ---------------------------------------------------------------------------
// PtaBuilder
// ---------------------------------------------------------------------------

/// Converts a `GuidelineDocument` into a PTA.
#[derive(Debug)]
pub struct PtaBuilder {
    clock_prefix: String,
    variable_prefix: String,
    edge_counter: usize,
    /// When true, generate monitoring locations for each MonitoringRequirement.
    pub include_monitoring_locations: bool,
    /// When true, generate safety-constraint guard conjuncts on relevant edges.
    pub enforce_safety_guards: bool,
}

impl Default for PtaBuilder {
    fn default() -> Self {
        Self {
            clock_prefix: "clk".into(),
            variable_prefix: "var".into(),
            edge_counter: 0,
            include_monitoring_locations: true,
            enforce_safety_guards: true,
        }
    }
}

impl PtaBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_clock_prefix(mut self, p: &str) -> Self {
        self.clock_prefix = p.to_string();
        self
    }

    pub fn with_variable_prefix(mut self, p: &str) -> Self {
        self.variable_prefix = p.to_string();
        self
    }

    /// Build a PTA from a guideline document.
    pub fn build(&mut self, doc: &GuidelineDocument) -> PTA {
        let mut pta = PTA::new(&doc.metadata.title);

        // Metadata
        pta.metadata
            .insert("source_guideline".into(), doc.id.clone());
        pta.metadata
            .insert("version".into(), doc.metadata.version.clone());

        // --- Locations from decision points ---
        for dp in &doc.decision_points {
            let loc = self.build_location(dp);
            if loc.is_initial {
                pta.initial_location = Some(loc.id.clone());
            }
            pta.locations.push(loc);
        }

        // --- Edges from branches ---
        for dp in &doc.decision_points {
            for br in &dp.branches {
                let edge = self.build_edge_from_branch(&dp.id, br, doc);
                // Collect clocks
                for c in edge.guard.clocks() {
                    if !pta.clocks.contains(&c) {
                        pta.clocks.push(c);
                    }
                }
                for r in &edge.resets {
                    if !pta.clocks.contains(&r.clock) {
                        pta.clocks.push(r.clock.clone());
                    }
                }
                pta.edges.push(edge);
            }
        }

        // --- Edges from explicit transitions ---
        for tr in &doc.transitions {
            let edge = self.build_edge_from_transition(tr, doc);
            for c in edge.guard.clocks() {
                if !pta.clocks.contains(&c) {
                    pta.clocks.push(c);
                }
            }
            pta.edges.push(edge);
        }

        // --- Variables from medication actions ---
        let meds = doc.all_medications();
        for med in &meds {
            pta.variables.push(PtaVariable {
                name: format!("{}_active", med),
                var_type: PtaVarType::Bool,
                initial_value: 0.0,
                min: Some(0.0),
                max: Some(1.0),
            });
            pta.variables.push(PtaVariable {
                name: format!("{}_dose", med),
                var_type: PtaVarType::Real,
                initial_value: 0.0,
                min: Some(0.0),
                max: None,
            });
        }

        // --- Monitoring locations ---
        if self.include_monitoring_locations {
            for mr in &doc.monitoring {
                let (loc, edges) = self.build_monitoring_location(mr, doc);
                if !pta.locations.iter().any(|l| l.id == loc.id) {
                    pta.locations.push(loc);
                }
                for edge in edges {
                    pta.edges.push(edge);
                }
                let monitor_clock = format!("{}_{}", self.clock_prefix, mr.parameter);
                if !pta.clocks.contains(&monitor_clock) {
                    pta.clocks.push(monitor_clock);
                }
            }
        }

        // --- Add a global treatment clock ---
        let global_clock = format!("{}_treatment", self.clock_prefix);
        if !pta.clocks.contains(&global_clock) {
            pta.clocks.push(global_clock);
        }

        // --- Enforce safety constraint guards ---
        if self.enforce_safety_guards {
            self.apply_safety_constraints(&mut pta, &doc.safety_constraints);
        }

        pta
    }

    // ----- helpers --------------------------------------------------------

    fn next_edge_id(&mut self) -> String {
        self.edge_counter += 1;
        format!("e_{}", self.edge_counter)
    }

    fn build_location(&self, dp: &DecisionPoint) -> Location {
        let invariants: Vec<Invariant> = dp
            .invariants
            .iter()
            .filter_map(|inv_str| self.parse_invariant_string(inv_str))
            .collect();

        Location {
            id: dp.id.clone(),
            name: dp.label.clone(),
            invariants,
            is_initial: dp.is_initial,
            is_urgent: dp.urgency.map_or(false, |u| {
                matches!(
                    u,
                    crate::format::Urgency::Urgent | crate::format::Urgency::Emergent
                )
            }),
            is_committed: false,
            cost_rate: None,
            labels: vec![],
        }
    }

    fn parse_invariant_string(&self, s: &str) -> Option<Invariant> {
        // Simple parser for "clock <= value" format
        let parts: Vec<&str> = s.split_whitespace().collect();
        if parts.len() != 3 {
            return None;
        }
        let clock = parts[0].to_string();
        let op = match parts[1] {
            "<" => ComparisonOp::Lt,
            "<=" => ComparisonOp::Le,
            "==" => ComparisonOp::Eq,
            ">=" => ComparisonOp::Ge,
            ">" => ComparisonOp::Gt,
            "!=" => ComparisonOp::Ne,
            _ => return None,
        };
        let bound: f64 = parts[2].parse().ok()?;
        Some(Invariant {
            clock,
            op,
            bound: OrderedFloat(bound),
        })
    }

    fn build_edge_from_branch(
        &mut self,
        source_id: &str,
        branch: &Branch,
        _doc: &GuidelineDocument,
    ) -> Edge {
        let guard = self.compile_guideline_guard(&branch.guard);
        let mut resets = Vec::new();
        let mut updates = Vec::new();

        for action in &branch.actions {
            let (r, u) = self.action_to_updates(action);
            resets.extend(r);
            updates.extend(u);
        }

        Edge {
            id: self.next_edge_id(),
            source: source_id.to_string(),
            target: branch.target.clone(),
            guard,
            resets,
            updates,
            sync: None,
            weight: None,
            label: Some(branch.id.clone()),
        }
    }

    fn build_edge_from_transition(
        &mut self,
        tr: &TransitionRule,
        _doc: &GuidelineDocument,
    ) -> Edge {
        let guard = self.compile_guideline_guard(&tr.guard);
        let mut resets = Vec::new();
        let mut updates = Vec::new();

        for action in &tr.actions {
            let (r, u) = self.action_to_updates(action);
            resets.extend(r);
            updates.extend(u);
        }

        Edge {
            id: self.next_edge_id(),
            source: tr.source.clone(),
            target: tr.target.clone(),
            guard,
            resets,
            updates,
            sync: None,
            weight: tr.weight,
            label: Some(tr.id.clone()),
        }
    }

    /// Convert a `GuidelineGuard` to a PTA `Guard`.
    pub fn compile_guideline_guard(&self, gg: &GuidelineGuard) -> Guard {
        match gg {
            GuidelineGuard::ClinicalPredicate {
                parameter,
                operator,
                threshold,
                ..
            } => Guard::VariableConstraint {
                variable: parameter.clone(),
                op: *operator,
                value: *threshold,
            },
            GuidelineGuard::LabThreshold {
                test_name,
                operator,
                value,
                ..
            } => Guard::VariableConstraint {
                variable: test_name.clone(),
                op: *operator,
                value: *value,
            },
            GuidelineGuard::TimeElapsed {
                clock,
                operator,
                days,
            } => Guard::ClockConstraint {
                clock: clock.clone(),
                op: *operator,
                value: *days,
            },
            GuidelineGuard::ConcentrationRange {
                drug, min, max, ..
            } => Guard::RangeConstraint {
                variable: format!("{}_concentration", drug),
                min: *min,
                max: *max,
            },
            GuidelineGuard::DiagnosisPresent { diagnosis } => Guard::BooleanVariable {
                variable: format!("dx_{}", diagnosis),
                expected: true,
            },
            GuidelineGuard::MedicationActive { medication } => Guard::BooleanVariable {
                variable: format!("{}_active", medication),
                expected: true,
            },
            GuidelineGuard::AgeRange {
                min_years,
                max_years,
            } => {
                let mut parts = Vec::new();
                if let Some(min) = min_years {
                    parts.push(Guard::VariableConstraint {
                        variable: "age".into(),
                        op: ComparisonOp::Ge,
                        value: *min,
                    });
                }
                if let Some(max) = max_years {
                    parts.push(Guard::VariableConstraint {
                        variable: "age".into(),
                        op: ComparisonOp::Le,
                        value: *max,
                    });
                }
                match parts.len() {
                    0 => Guard::True,
                    1 => parts.into_iter().next().unwrap(),
                    _ => Guard::And(parts),
                }
            }
            GuidelineGuard::AllergyPresent { substance } => Guard::BooleanVariable {
                variable: format!("allergy_{}", substance),
                expected: true,
            },
            GuidelineGuard::And(gs) => {
                let compiled: Vec<Guard> =
                    gs.iter().map(|g| self.compile_guideline_guard(g)).collect();
                if compiled.is_empty() {
                    Guard::True
                } else if compiled.len() == 1 {
                    compiled.into_iter().next().unwrap()
                } else {
                    Guard::And(compiled)
                }
            }
            GuidelineGuard::Or(gs) => {
                let compiled: Vec<Guard> =
                    gs.iter().map(|g| self.compile_guideline_guard(g)).collect();
                if compiled.is_empty() {
                    Guard::False
                } else if compiled.len() == 1 {
                    compiled.into_iter().next().unwrap()
                } else {
                    Guard::Or(compiled)
                }
            }
            GuidelineGuard::Not(g) => {
                let inner = self.compile_guideline_guard(g);
                inner.negate()
            }
            GuidelineGuard::True => Guard::True,
            GuidelineGuard::False => Guard::False,
        }
    }

    /// Convert an action into clock resets and variable updates.
    fn action_to_updates(&self, action: &GuidelineAction) -> (Vec<Reset>, Vec<Update>) {
        let mut resets = Vec::new();
        let mut updates = Vec::new();

        match action {
            GuidelineAction::StartMedication {
                medication, dose, ..
            } => {
                updates.push(Update {
                    variable: format!("{}_active", medication),
                    expression: UpdateExpr::SetBool(true),
                });
                updates.push(Update {
                    variable: format!("{}_dose", medication),
                    expression: UpdateExpr::Assign(dose.value),
                });
                resets.push(Reset {
                    clock: format!("{}_{}", self.clock_prefix, medication),
                    value: 0.0,
                });
            }
            GuidelineAction::AdjustDose {
                medication,
                new_dose,
                ..
            } => {
                updates.push(Update {
                    variable: format!("{}_dose", medication),
                    expression: UpdateExpr::Assign(new_dose.value),
                });
            }
            GuidelineAction::StopMedication { medication, .. } => {
                updates.push(Update {
                    variable: format!("{}_active", medication),
                    expression: UpdateExpr::SetBool(false),
                });
                updates.push(Update {
                    variable: format!("{}_dose", medication),
                    expression: UpdateExpr::Assign(0.0),
                });
            }
            GuidelineAction::SwitchMedication {
                from_medication,
                to_medication,
                new_dose,
                ..
            } => {
                updates.push(Update {
                    variable: format!("{}_active", from_medication),
                    expression: UpdateExpr::SetBool(false),
                });
                updates.push(Update {
                    variable: format!("{}_dose", from_medication),
                    expression: UpdateExpr::Assign(0.0),
                });
                updates.push(Update {
                    variable: format!("{}_active", to_medication),
                    expression: UpdateExpr::SetBool(true),
                });
                updates.push(Update {
                    variable: format!("{}_dose", to_medication),
                    expression: UpdateExpr::Assign(new_dose.value),
                });
                resets.push(Reset {
                    clock: format!("{}_{}", self.clock_prefix, to_medication),
                    value: 0.0,
                });
            }
            GuidelineAction::SetClock { clock_name, value } => {
                resets.push(Reset {
                    clock: clock_name.clone(),
                    value: *value,
                });
            }
            GuidelineAction::ResetClock { clock_name } => {
                resets.push(Reset {
                    clock: clock_name.clone(),
                    value: 0.0,
                });
            }
            GuidelineAction::CombinationTherapy { medications, .. } => {
                for med in medications {
                    updates.push(Update {
                        variable: format!("{}_active", med.name),
                        expression: UpdateExpr::SetBool(true),
                    });
                    updates.push(Update {
                        variable: format!("{}_dose", med.name),
                        expression: UpdateExpr::Assign(med.dose.value),
                    });
                    resets.push(Reset {
                        clock: format!("{}_{}", self.clock_prefix, med.name),
                        value: 0.0,
                    });
                }
            }
            GuidelineAction::EmergencyIntervention { medications, .. } => {
                for med in medications {
                    updates.push(Update {
                        variable: format!("{}_active", med.name),
                        expression: UpdateExpr::SetBool(true),
                    });
                    updates.push(Update {
                        variable: format!("{}_dose", med.name),
                        expression: UpdateExpr::Assign(med.dose.value),
                    });
                }
            }
            GuidelineAction::MonitorInterval {
                parameter,
                interval_days,
                ..
            } => {
                resets.push(Reset {
                    clock: format!("{}_{}", self.clock_prefix, parameter),
                    value: 0.0,
                });
            }
            _ => {
                // Other actions (OrderLab, Refer, Lifestyle, etc.) don't
                // directly map to PTA clock/variable updates.
            }
        }

        (resets, updates)
    }

    fn build_monitoring_location(
        &mut self,
        mr: &MonitoringRequirement,
        doc: &GuidelineDocument,
    ) -> (Location, Vec<Edge>) {
        let loc_id = format!("monitor_{}", mr.parameter);
        let monitor_clock = format!("{}_{}", self.clock_prefix, mr.parameter);

        let invariant = Invariant {
            clock: monitor_clock.clone(),
            op: ComparisonOp::Le,
            bound: OrderedFloat(mr.interval_days as f64),
        };

        let location = Location {
            id: loc_id.clone(),
            name: format!("Monitor {}", mr.parameter),
            invariants: vec![invariant],
            is_initial: false,
            is_urgent: false,
            is_committed: false,
            cost_rate: None,
            labels: vec![format!("monitoring:{}", mr.parameter)],
        };

        let mut edges = Vec::new();

        // For each state the monitoring applies to (or all states if empty),
        // add an edge from that state to the monitoring location and back.
        let applicable_states: Vec<String> = if mr.applies_to_states.is_empty() {
            doc.decision_points
                .iter()
                .filter(|dp| !dp.is_terminal)
                .map(|dp| dp.id.clone())
                .collect()
        } else {
            mr.applies_to_states.clone()
        };

        for state in &applicable_states {
            // Edge to monitoring location
            let to_monitor = Edge {
                id: self.next_edge_id(),
                source: state.clone(),
                target: loc_id.clone(),
                guard: Guard::ClockConstraint {
                    clock: monitor_clock.clone(),
                    op: ComparisonOp::Ge,
                    value: mr.interval_days as f64,
                },
                resets: vec![],
                updates: vec![],
                sync: Some(format!("check_{}", mr.parameter)),
                weight: None,
                label: Some(format!("monitor_trigger_{}", mr.parameter)),
            };

            // Edge back from monitoring location
            let from_monitor = Edge {
                id: self.next_edge_id(),
                source: loc_id.clone(),
                target: state.clone(),
                guard: Guard::True,
                resets: vec![Reset {
                    clock: monitor_clock.clone(),
                    value: 0.0,
                }],
                updates: vec![],
                sync: Some(format!("done_{}", mr.parameter)),
                weight: None,
                label: Some(format!("monitor_return_{}", mr.parameter)),
            };

            edges.push(to_monitor);
            edges.push(from_monitor);
        }

        (location, edges)
    }

    fn apply_safety_constraints(&self, pta: &mut PTA, constraints: &[SafetyConstraint]) {
        for sc in constraints {
            let safety_guard = self.compile_guideline_guard(&sc.guard);
            let negated = safety_guard.negate();

            let applicable: Vec<String> = if sc.applies_to.is_empty() {
                pta.edges.iter().map(|e| e.id.clone()).collect()
            } else {
                pta.edges
                    .iter()
                    .filter(|e| {
                        sc.applies_to.contains(&e.source) || sc.applies_to.contains(&e.target)
                    })
                    .map(|e| e.id.clone())
                    .collect()
            };

            for edge in &mut pta.edges {
                if applicable.contains(&edge.id) {
                    edge.guard = edge.guard.clone().and(negated.clone());
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// PTA serialisation helpers
// ---------------------------------------------------------------------------

/// Serialise the PTA to JSON.
pub fn pta_to_json(pta: &PTA) -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(pta)
}

/// Serialise the PTA to a compact DOT (Graphviz) representation.
pub fn pta_to_dot(pta: &PTA) -> String {
    let mut out = String::new();
    out.push_str(&format!("digraph \"{}\" {{\n", pta.name));
    out.push_str("  rankdir=LR;\n");
    out.push_str("  node [shape=ellipse];\n\n");

    for loc in &pta.locations {
        let mut attrs = vec![format!("label=\"{}\"", loc.name)];
        if loc.is_initial {
            attrs.push("style=bold".into());
        }
        if loc.is_urgent {
            attrs.push("color=red".into());
        }
        out.push_str(&format!("  \"{}\" [{}];\n", loc.id, attrs.join(", ")));
    }

    out.push('\n');

    for edge in &pta.edges {
        let label = edge.label.clone().unwrap_or_else(|| edge.id.clone());
        out.push_str(&format!(
            "  \"{}\" -> \"{}\" [label=\"{}\"];\n",
            edge.source, edge.target, label
        ));
    }

    out.push_str("}\n");
    out
}

/// Collect statistics about a PTA.
pub fn pta_stats(pta: &PTA) -> HashMap<String, usize> {
    let mut stats = HashMap::new();
    stats.insert("locations".into(), pta.locations.len());
    stats.insert("edges".into(), pta.edges.len());
    stats.insert("clocks".into(), pta.clocks.len());
    stats.insert("variables".into(), pta.variables.len());
    stats.insert("unreachable".into(), pta.unreachable_count());
    let max_depth = pta.edges.iter().map(|e| e.guard.depth()).max().unwrap_or(0);
    stats.insert("max_guard_depth".into(), max_depth);
    stats
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{standard_diabetes_template, standard_hypertension_template};

    #[test]
    fn test_build_diabetes_pta() {
        let doc = standard_diabetes_template();
        let mut builder = PtaBuilder::new();
        let pta = builder.build(&doc);

        assert!(!pta.locations.is_empty());
        assert!(!pta.edges.is_empty());
        assert!(pta.initial_location.is_some());
        assert!(!pta.clocks.is_empty());
    }

    #[test]
    fn test_build_hypertension_pta() {
        let doc = standard_hypertension_template();
        let mut builder = PtaBuilder::new();
        let pta = builder.build(&doc);

        assert!(pta.locations.len() >= 4);
        assert!(!pta.edges.is_empty());
    }

    #[test]
    fn test_pta_reachability() {
        let doc = standard_diabetes_template();
        let mut builder = PtaBuilder::new();
        let pta = builder.build(&doc);
        let reachable = pta.reachable_locations();
        // Initial location should be reachable
        assert!(reachable.contains(pta.initial_location.as_ref().unwrap()));
    }

    #[test]
    fn test_guard_compilation() {
        let builder = PtaBuilder::new();
        let gg = GuidelineGuard::And(vec![
            GuidelineGuard::LabThreshold {
                test_name: "HbA1c".into(),
                operator: ComparisonOp::Ge,
                value: 7.0,
                unit: None,
            },
            GuidelineGuard::TimeElapsed {
                clock: "treatment_clock".into(),
                operator: ComparisonOp::Ge,
                days: 90.0,
            },
        ]);
        let guard = builder.compile_guideline_guard(&gg);
        match &guard {
            Guard::And(parts) => {
                assert_eq!(parts.len(), 2);
            }
            _ => panic!("Expected And guard"),
        }
        assert!(guard.clocks().contains("treatment_clock"));
        assert!(guard.variables().contains("HbA1c"));
    }

    #[test]
    fn test_medication_variables() {
        let doc = standard_diabetes_template();
        let mut builder = PtaBuilder::new();
        let pta = builder.build(&doc);
        let var_names: Vec<&str> = pta.variables.iter().map(|v| v.name.as_str()).collect();
        assert!(var_names.contains(&"metformin_active"));
        assert!(var_names.contains(&"metformin_dose"));
    }

    #[test]
    fn test_pta_to_json() {
        let doc = standard_diabetes_template();
        let mut builder = PtaBuilder::new();
        let pta = builder.build(&doc);
        let json = pta_to_json(&pta).unwrap();
        assert!(json.contains("locations"));
        assert!(json.contains("edges"));
    }

    #[test]
    fn test_pta_to_dot() {
        let doc = standard_diabetes_template();
        let mut builder = PtaBuilder::new();
        let pta = builder.build(&doc);
        let dot = pta_to_dot(&pta);
        assert!(dot.starts_with("digraph"));
        assert!(dot.contains("->"));
    }

    #[test]
    fn test_pta_stats() {
        let doc = standard_diabetes_template();
        let mut builder = PtaBuilder::new();
        let pta = builder.build(&doc);
        let stats = pta_stats(&pta);
        assert!(*stats.get("locations").unwrap() > 0);
        assert!(*stats.get("edges").unwrap() > 0);
    }

    #[test]
    fn test_guard_and_or_true_false() {
        let a = Guard::True;
        let b = Guard::VariableConstraint {
            variable: "x".into(),
            op: ComparisonOp::Gt,
            value: 5.0,
        };
        assert_eq!(a.and(b.clone()), b);

        let c = Guard::False;
        assert_eq!(c.or(b.clone()), b);
    }

    #[test]
    fn test_edge_from_to() {
        let doc = standard_diabetes_template();
        let mut builder = PtaBuilder::new();
        let pta = builder.build(&doc);
        let init = pta.initial_location.as_ref().unwrap();
        let from_init = pta.edges_from(init);
        assert!(!from_init.is_empty());
    }

    #[test]
    fn test_no_monitoring_locations_option() {
        let doc = standard_diabetes_template();
        let mut builder = PtaBuilder::new();
        builder.include_monitoring_locations = false;
        let pta = builder.build(&doc);
        // Should have no monitor_* locations
        assert!(pta
            .locations
            .iter()
            .all(|l| !l.id.starts_with("monitor_")));
    }

    #[test]
    fn test_safety_constraint_application() {
        let doc = standard_diabetes_template();
        let mut builder = PtaBuilder::new();
        builder.enforce_safety_guards = true;
        let pta = builder.build(&doc);
        // Safety constraints should cause some guards to contain Not(...)
        let has_negated = pta.edges.iter().any(|e| matches!(&e.guard, Guard::And(parts) if parts.iter().any(|p| matches!(p, Guard::Not(_)))));
        assert!(has_negated, "Expected at least one edge with safety-negated guard");
    }
}
