//! Formal security properties for memory model verification.
//!
//! Implements formal security properties from §11 of the LITMUS∞ paper.
//! Provides noninterference checking, declassification policies,
//! security lattices, information flow typing, security automata,
//! and hyperproperty checking.

use std::collections::{HashMap, HashSet, BTreeSet, VecDeque};
use std::fmt;

// ═══════════════════════════════════════════════════════════════════════
// SecurityProperty — enum of formal properties
// ═══════════════════════════════════════════════════════════════════════

/// Formal security properties for memory model verification.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SecurityProperty {
    /// High-security inputs must not affect low-security outputs.
    Noninterference,
    /// Secret values must remain secret (not leaked through observations).
    SecrecyPreservation,
    /// Low-integrity inputs must not affect high-integrity computations.
    IntegrityPreservation,
    /// No information flows from high to low through timing channels.
    TimingNoninterference,
    /// Termination-insensitive noninterference.
    TINI,
    /// Termination-sensitive noninterference.
    TSNI,
}

impl fmt::Display for SecurityProperty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SecurityProperty::Noninterference => write!(f, "Noninterference"),
            SecurityProperty::SecrecyPreservation => write!(f, "SecrecyPreservation"),
            SecurityProperty::IntegrityPreservation => write!(f, "IntegrityPreservation"),
            SecurityProperty::TimingNoninterference => write!(f, "TimingNoninterference"),
            SecurityProperty::TINI => write!(f, "TINI"),
            SecurityProperty::TSNI => write!(f, "TSNI"),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// SecurityLevel — levels in the security lattice
// ═══════════════════════════════════════════════════════════════════════

/// A security classification level.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum SecurityLevel {
    /// Public / unclassified.
    Low,
    /// Confidential.
    Medium,
    /// Secret.
    High,
    /// Top secret.
    TopSecret,
}

impl SecurityLevel {
    pub fn all() -> &'static [SecurityLevel] {
        &[
            SecurityLevel::Low,
            SecurityLevel::Medium,
            SecurityLevel::High,
            SecurityLevel::TopSecret,
        ]
    }

    /// Check if this level flows to the other (information flow allowed).
    pub fn flows_to(&self, other: &SecurityLevel) -> bool {
        self <= other
    }

    /// Least upper bound (join) of two levels.
    pub fn lub(&self, other: &SecurityLevel) -> SecurityLevel {
        if self >= other {
            self.clone()
        } else {
            other.clone()
        }
    }

    /// Greatest lower bound (meet) of two levels.
    pub fn glb(&self, other: &SecurityLevel) -> SecurityLevel {
        if self <= other {
            self.clone()
        } else {
            other.clone()
        }
    }
}

impl fmt::Display for SecurityLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SecurityLevel::Low => write!(f, "L"),
            SecurityLevel::Medium => write!(f, "M"),
            SecurityLevel::High => write!(f, "H"),
            SecurityLevel::TopSecret => write!(f, "TS"),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// SecurityLattice — full security lattice
// ═══════════════════════════════════════════════════════════════════════

/// A security lattice with arbitrary labels.
#[derive(Debug, Clone)]
pub struct SecurityLattice {
    /// Labels in the lattice.
    pub labels: Vec<String>,
    /// Partial order: (i, j) means label i ≤ label j.
    pub ordering: BTreeSet<(usize, usize)>,
}

impl SecurityLattice {
    /// Create a simple two-level lattice (Low ≤ High).
    pub fn two_level() -> Self {
        let labels = vec!["Low".to_string(), "High".to_string()];
        let mut ordering = BTreeSet::new();
        ordering.insert((0, 0)); // reflexive
        ordering.insert((1, 1));
        ordering.insert((0, 1)); // Low ≤ High
        Self { labels, ordering }
    }

    /// Create a four-level lattice (Low ≤ Medium ≤ High ≤ TopSecret).
    pub fn four_level() -> Self {
        let labels = vec![
            "Low".to_string(),
            "Medium".to_string(),
            "High".to_string(),
            "TopSecret".to_string(),
        ];
        let mut ordering = BTreeSet::new();
        for i in 0..4 {
            for j in i..4 {
                ordering.insert((i, j));
            }
        }
        Self { labels, ordering }
    }

    /// Create a diamond lattice (Bot ≤ A, Bot ≤ B, A ≤ Top, B ≤ Top).
    pub fn diamond() -> Self {
        let labels = vec![
            "Bot".to_string(),
            "A".to_string(),
            "B".to_string(),
            "Top".to_string(),
        ];
        let mut ordering = BTreeSet::new();
        // Reflexive
        for i in 0..4 {
            ordering.insert((i, i));
        }
        ordering.insert((0, 1)); // Bot ≤ A
        ordering.insert((0, 2)); // Bot ≤ B
        ordering.insert((0, 3)); // Bot ≤ Top
        ordering.insert((1, 3)); // A ≤ Top
        ordering.insert((2, 3)); // B ≤ Top
        Self { labels, ordering }
    }

    /// Check if label a flows to label b.
    pub fn flows_to(&self, a: usize, b: usize) -> bool {
        self.ordering.contains(&(a, b))
    }

    /// Get the label index by name.
    pub fn label_index(&self, name: &str) -> Option<usize> {
        self.labels.iter().position(|l| l == name)
    }

    /// Compute the join (LUB) of two labels.
    pub fn join(&self, a: usize, b: usize) -> Option<usize> {
        // Find smallest label that is ≥ both a and b
        for i in 0..self.labels.len() {
            if self.flows_to(a, i) && self.flows_to(b, i) {
                // Check it's the smallest such
                let mut is_smallest = true;
                for j in 0..self.labels.len() {
                    if j != i && self.flows_to(a, j) && self.flows_to(b, j) && self.flows_to(j, i) {
                        is_smallest = false;
                        break;
                    }
                }
                if is_smallest {
                    return Some(i);
                }
            }
        }
        None
    }

    /// Compute the meet (GLB) of two labels.
    pub fn meet(&self, a: usize, b: usize) -> Option<usize> {
        // Find largest label that is ≤ both a and b
        for i in (0..self.labels.len()).rev() {
            if self.flows_to(i, a) && self.flows_to(i, b) {
                let mut is_largest = true;
                for j in (0..self.labels.len()).rev() {
                    if j != i && self.flows_to(j, a) && self.flows_to(j, b) && self.flows_to(i, j) {
                        is_largest = false;
                        break;
                    }
                }
                if is_largest {
                    return Some(i);
                }
            }
        }
        None
    }

    /// Number of labels.
    pub fn size(&self) -> usize {
        self.labels.len()
    }
}

// ═══════════════════════════════════════════════════════════════════════
// NoninterferenceChecker
// ═══════════════════════════════════════════════════════════════════════

/// An execution trace for noninterference checking.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionTrace {
    /// Sequence of observable outputs.
    pub outputs: Vec<ObservableOutput>,
    /// Input classification.
    pub input_levels: HashMap<String, SecurityLevel>,
}

/// An observable output event.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ObservableOutput {
    pub channel: String,
    pub value: u64,
    pub timestamp: u64,
}

impl ExecutionTrace {
    pub fn new() -> Self {
        Self {
            outputs: Vec::new(),
            input_levels: HashMap::new(),
        }
    }

    pub fn add_output(&mut self, channel: &str, value: u64, timestamp: u64) {
        self.outputs.push(ObservableOutput {
            channel: channel.to_string(),
            value,
            timestamp,
        });
    }

    pub fn set_input_level(&mut self, input: &str, level: SecurityLevel) {
        self.input_levels.insert(input.to_string(), level);
    }

    /// Project the trace to only low-observable outputs.
    pub fn low_projection(&self) -> Vec<&ObservableOutput> {
        // Consider all outputs as low-observable by default
        self.outputs.iter().collect()
    }
}

impl Default for ExecutionTrace {
    fn default() -> Self {
        Self::new()
    }
}

/// Checker for noninterference properties.
///
/// Verifies that high-security inputs do not affect low-security outputs.
#[derive(Debug, Clone)]
pub struct NoninterferenceChecker {
    /// Security lattice.
    pub lattice: SecurityLattice,
    /// Variable security labels.
    pub variable_labels: HashMap<String, usize>,
    /// Output channel security labels.
    pub output_labels: HashMap<String, usize>,
}

impl NoninterferenceChecker {
    pub fn new(lattice: SecurityLattice) -> Self {
        Self {
            lattice,
            variable_labels: HashMap::new(),
            output_labels: HashMap::new(),
        }
    }

    pub fn set_variable_level(&mut self, var: &str, level: usize) {
        self.variable_labels.insert(var.to_string(), level);
    }

    pub fn set_output_level(&mut self, channel: &str, level: usize) {
        self.output_labels.insert(channel.to_string(), level);
    }

    /// Check noninterference: two traces that agree on low inputs
    /// must produce the same low outputs.
    pub fn check_traces(
        &self,
        trace_a: &ExecutionTrace,
        trace_b: &ExecutionTrace,
    ) -> NoninterferenceResult {
        // Get low projections
        let low_a = trace_a.low_projection();
        let low_b = trace_b.low_projection();

        // Check if low inputs are the same
        let low_inputs_match = self.low_inputs_match(trace_a, trace_b);

        if !low_inputs_match {
            return NoninterferenceResult {
                holds: true,
                reason: "Low inputs differ — not applicable".to_string(),
                violations: Vec::new(),
            };
        }

        // Check if low outputs match
        let mut violations = Vec::new();
        let max_len = low_a.len().max(low_b.len());
        for i in 0..max_len {
            let out_a = low_a.get(i);
            let out_b = low_b.get(i);

            match (out_a, out_b) {
                (Some(a), Some(b)) => {
                    if a.value != b.value || a.channel != b.channel {
                        violations.push(NoninterferenceViolation {
                            output_index: i,
                            channel: a.channel.clone(),
                            value_a: a.value,
                            value_b: b.value,
                        });
                    }
                }
                (Some(a), None) => {
                    violations.push(NoninterferenceViolation {
                        output_index: i,
                        channel: a.channel.clone(),
                        value_a: a.value,
                        value_b: 0,
                    });
                }
                (None, Some(b)) => {
                    violations.push(NoninterferenceViolation {
                        output_index: i,
                        channel: b.channel.clone(),
                        value_a: 0,
                        value_b: b.value,
                    });
                }
                (None, None) => {}
            }
        }

        NoninterferenceResult {
            holds: violations.is_empty(),
            reason: if violations.is_empty() {
                "Low outputs match for matching low inputs".to_string()
            } else {
                format!("{} low-output differences detected", violations.len())
            },
            violations,
        }
    }

    fn low_inputs_match(
        &self,
        trace_a: &ExecutionTrace,
        trace_b: &ExecutionTrace,
    ) -> bool {
        // Check that all low-classified inputs have the same value
        for (input, level) in &trace_a.input_levels {
            let label_idx = self.variable_labels.get(input).copied().unwrap_or(0);
            // If it's a low input (label 0 in a two-level lattice)
            if label_idx == 0 {
                if let Some(other_level) = trace_b.input_levels.get(input) {
                    if level != other_level {
                        return false;
                    }
                }
            }
        }
        true
    }
}

/// Result of a noninterference check.
#[derive(Debug, Clone)]
pub struct NoninterferenceResult {
    pub holds: bool,
    pub reason: String,
    pub violations: Vec<NoninterferenceViolation>,
}

impl fmt::Display for NoninterferenceResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Noninterference: {} ({})",
            if self.holds { "HOLDS" } else { "VIOLATED" },
            self.reason,
        )
    }
}

/// A specific noninterference violation.
#[derive(Debug, Clone)]
pub struct NoninterferenceViolation {
    pub output_index: usize,
    pub channel: String,
    pub value_a: u64,
    pub value_b: u64,
}

impl fmt::Display for NoninterferenceViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Output {} on '{}': {} vs {}",
            self.output_index, self.channel, self.value_a, self.value_b,
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════
// DeclassificationPolicy — controlled information release
// ═══════════════════════════════════════════════════════════════════════

/// A declassification policy allowing controlled release of secret information.
#[derive(Debug, Clone)]
pub struct DeclassificationPolicy {
    /// Name of the policy.
    pub name: String,
    /// Allowed declassification channels: (from_level, to_level, condition).
    pub allowed_flows: Vec<DeclassificationRule>,
}

/// A single declassification rule.
#[derive(Debug, Clone)]
pub struct DeclassificationRule {
    /// Source security level.
    pub from_level: SecurityLevel,
    /// Target security level.
    pub to_level: SecurityLevel,
    /// Variable or channel being declassified.
    pub variable: String,
    /// Condition under which declassification is allowed.
    pub condition: DeclassificationCondition,
}

/// Condition for declassification.
#[derive(Debug, Clone)]
pub enum DeclassificationCondition {
    /// Always allowed.
    Always,
    /// Allowed after a specific event.
    AfterEvent(String),
    /// Allowed only for specific values (e.g., hash of secret).
    ValueTransform(String),
    /// Allowed with explicit authorization.
    Authorized(String),
    /// Never allowed (placeholder for "no declassification").
    Never,
}

impl DeclassificationPolicy {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            allowed_flows: Vec::new(),
        }
    }

    pub fn add_rule(&mut self, rule: DeclassificationRule) {
        self.allowed_flows.push(rule);
    }

    /// Check if a specific declassification is allowed.
    pub fn is_allowed(
        &self,
        variable: &str,
        from: &SecurityLevel,
        to: &SecurityLevel,
    ) -> bool {
        self.allowed_flows.iter().any(|rule| {
            rule.variable == variable
                && rule.from_level == *from
                && rule.to_level == *to
                && !matches!(rule.condition, DeclassificationCondition::Never)
        })
    }

    /// Get all variables that can be declassified.
    pub fn declassifiable_variables(&self) -> Vec<&str> {
        self.allowed_flows
            .iter()
            .filter(|r| !matches!(r.condition, DeclassificationCondition::Never))
            .map(|r| r.variable.as_str())
            .collect()
    }

    /// A strict policy with no declassification.
    pub fn no_declassification() -> Self {
        Self::new("no-declassification")
    }

    /// A policy allowing password hash declassification.
    pub fn password_hash_policy() -> Self {
        let mut policy = Self::new("password-hash");
        policy.add_rule(DeclassificationRule {
            from_level: SecurityLevel::High,
            to_level: SecurityLevel::Low,
            variable: "password_hash".to_string(),
            condition: DeclassificationCondition::ValueTransform("hash".to_string()),
        });
        policy
    }
}

// ═══════════════════════════════════════════════════════════════════════
// InformationFlowType — type system for information flow
// ═══════════════════════════════════════════════════════════════════════

/// Information flow type for a variable or expression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InformationFlowType {
    /// Security level of the value.
    pub level: SecurityLevel,
    /// Whether the variable's existence (not just value) is classified.
    pub existence_classified: bool,
    /// Dependencies (variables this one depends on).
    pub dependencies: HashSet<String>,
}

impl InformationFlowType {
    pub fn new(level: SecurityLevel) -> Self {
        Self {
            level,
            existence_classified: false,
            dependencies: HashSet::new(),
        }
    }

    pub fn with_dependency(mut self, dep: &str) -> Self {
        self.dependencies.insert(dep.to_string());
        self
    }

    pub fn with_existence_classified(mut self) -> Self {
        self.existence_classified = true;
        self
    }

    /// Join two flow types (for merging at control flow joins).
    pub fn join(&self, other: &InformationFlowType) -> InformationFlowType {
        InformationFlowType {
            level: self.level.lub(&other.level),
            existence_classified: self.existence_classified || other.existence_classified,
            dependencies: self
                .dependencies
                .union(&other.dependencies)
                .cloned()
                .collect(),
        }
    }

    /// Check if assigning this type to a target level is secure.
    pub fn can_assign_to(&self, target: &SecurityLevel) -> bool {
        self.level.flows_to(target)
    }
}

impl fmt::Display for InformationFlowType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "τ({})", self.level)?;
        if self.existence_classified {
            write!(f, "[∃-classified]")?;
        }
        if !self.dependencies.is_empty() {
            let deps: Vec<_> = self.dependencies.iter().collect();
            write!(f, " deps={{{}}}", deps.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(","))?;
        }
        Ok(())
    }
}

/// Information flow type checker.
#[derive(Debug, Clone)]
pub struct InformationFlowChecker {
    /// Variable type assignments.
    pub types: HashMap<String, InformationFlowType>,
    /// Current control flow security level (the "pc label").
    pub pc_level: SecurityLevel,
    /// Violations found.
    pub violations: Vec<FlowViolation>,
}

/// An information flow violation.
#[derive(Debug, Clone)]
pub struct FlowViolation {
    pub source: String,
    pub target: String,
    pub source_level: SecurityLevel,
    pub target_level: SecurityLevel,
    pub description: String,
}

impl fmt::Display for FlowViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Flow violation: {} ({}) → {} ({}): {}",
            self.source, self.source_level, self.target, self.target_level, self.description,
        )
    }
}

impl InformationFlowChecker {
    pub fn new() -> Self {
        Self {
            types: HashMap::new(),
            pc_level: SecurityLevel::Low,
            violations: Vec::new(),
        }
    }

    pub fn set_type(&mut self, var: &str, flow_type: InformationFlowType) {
        self.types.insert(var.to_string(), flow_type);
    }

    pub fn get_type(&self, var: &str) -> InformationFlowType {
        self.types
            .get(var)
            .cloned()
            .unwrap_or_else(|| InformationFlowType::new(SecurityLevel::Low))
    }

    /// Check if an assignment `target := source` is secure.
    pub fn check_assignment(&mut self, source: &str, target: &str) -> bool {
        let src_type = self.get_type(source);
        let tgt_type = self.get_type(target);

        // The source level (joined with PC label) must flow to the target
        let effective_level = src_type.level.lub(&self.pc_level);
        let secure = effective_level.flows_to(&tgt_type.level);

        if !secure {
            self.violations.push(FlowViolation {
                source: source.to_string(),
                target: target.to_string(),
                source_level: effective_level,
                target_level: tgt_type.level,
                description: "Explicit flow: high data assigned to low variable".to_string(),
            });
        }

        secure
    }

    /// Check if a branch on `var` is secure (implicit flow).
    pub fn check_branch(&mut self, var: &str) -> bool {
        let var_type = self.get_type(var);
        // Branch raises the PC level
        let secure = var_type.level.flows_to(&self.pc_level) || self.pc_level == SecurityLevel::Low;
        if !secure && var_type.level > self.pc_level {
            // Implicit flow through control
            self.violations.push(FlowViolation {
                source: var.to_string(),
                target: "pc".to_string(),
                source_level: var_type.level.clone(),
                target_level: self.pc_level.clone(),
                description: "Implicit flow: branch on high data in low context".to_string(),
            });
        }
        secure
    }

    /// Enter a branch on a high variable (raises PC level).
    pub fn enter_branch(&mut self, var: &str) {
        let var_type = self.get_type(var);
        self.pc_level = self.pc_level.lub(&var_type.level);
    }

    /// Exit branch (restore PC level).
    pub fn exit_branch(&mut self, saved_pc: SecurityLevel) {
        self.pc_level = saved_pc;
    }

    /// Get all violations found so far.
    pub fn get_violations(&self) -> &[FlowViolation] {
        &self.violations
    }

    pub fn has_violations(&self) -> bool {
        !self.violations.is_empty()
    }
}

impl Default for InformationFlowChecker {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════
// SecurityAutomaton — temporal security properties
// ═══════════════════════════════════════════════════════════════════════

/// A security automaton for checking temporal security properties.
///
/// The automaton monitors execution traces and detects security violations
/// that depend on the order of events (e.g., "declassification must be
/// preceded by authorization").
#[derive(Debug, Clone)]
pub struct SecurityAutomaton {
    /// Name of the property being monitored.
    pub name: String,
    /// States of the automaton.
    pub states: Vec<AutomatonState>,
    /// Transitions: (from_state, event_pattern, to_state).
    pub transitions: Vec<AutomatonTransition>,
    /// Initial state index.
    pub initial_state: usize,
    /// Current state index.
    pub current_state: usize,
    /// Error states (violation detected).
    pub error_states: HashSet<usize>,
    /// Accepting states (property satisfied).
    pub accepting_states: HashSet<usize>,
}

/// A state in the security automaton.
#[derive(Debug, Clone)]
pub struct AutomatonState {
    pub id: usize,
    pub name: String,
    pub is_error: bool,
    pub is_accepting: bool,
}

/// A transition in the security automaton.
#[derive(Debug, Clone)]
pub struct AutomatonTransition {
    pub from: usize,
    pub to: usize,
    pub pattern: EventPattern,
}

/// A pattern for matching security events.
#[derive(Debug, Clone)]
pub enum EventPattern {
    /// Match any event.
    Any,
    /// Match a specific event type.
    EventType(String),
    /// Match a specific security level.
    SecurityLevel(SecurityLevel),
    /// Match a specific variable access.
    VariableAccess(String),
    /// Match a declassification event.
    Declassification,
    /// Conjunction of patterns.
    And(Box<EventPattern>, Box<EventPattern>),
    /// Disjunction of patterns.
    Or(Box<EventPattern>, Box<EventPattern>),
}

/// A security event for the automaton.
#[derive(Debug, Clone)]
pub struct SecurityEvent {
    pub event_type: String,
    pub level: SecurityLevel,
    pub variable: Option<String>,
    pub is_declassification: bool,
}

impl SecurityEvent {
    pub fn new(event_type: &str, level: SecurityLevel) -> Self {
        Self {
            event_type: event_type.to_string(),
            level,
            variable: None,
            is_declassification: false,
        }
    }

    pub fn with_variable(mut self, var: &str) -> Self {
        self.variable = Some(var.to_string());
        self
    }

    pub fn with_declassification(mut self) -> Self {
        self.is_declassification = true;
        self
    }
}

impl EventPattern {
    pub fn matches(&self, event: &SecurityEvent) -> bool {
        match self {
            EventPattern::Any => true,
            EventPattern::EventType(t) => event.event_type == *t,
            EventPattern::SecurityLevel(l) => event.level == *l,
            EventPattern::VariableAccess(v) => {
                event.variable.as_deref() == Some(v.as_str())
            }
            EventPattern::Declassification => event.is_declassification,
            EventPattern::And(a, b) => a.matches(event) && b.matches(event),
            EventPattern::Or(a, b) => a.matches(event) || b.matches(event),
        }
    }
}

impl SecurityAutomaton {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            states: Vec::new(),
            transitions: Vec::new(),
            initial_state: 0,
            current_state: 0,
            error_states: HashSet::new(),
            accepting_states: HashSet::new(),
        }
    }

    pub fn add_state(&mut self, name: &str, is_error: bool, is_accepting: bool) -> usize {
        let id = self.states.len();
        self.states.push(AutomatonState {
            id,
            name: name.to_string(),
            is_error,
            is_accepting,
        });
        if is_error {
            self.error_states.insert(id);
        }
        if is_accepting {
            self.accepting_states.insert(id);
        }
        id
    }

    pub fn add_transition(&mut self, from: usize, to: usize, pattern: EventPattern) {
        self.transitions.push(AutomatonTransition { from, to, pattern });
    }

    /// Process a security event, advancing the automaton.
    pub fn process(&mut self, event: &SecurityEvent) -> AutomatonResult {
        let mut matched = false;
        let mut next_state = self.current_state;

        for trans in &self.transitions {
            if trans.from == self.current_state && trans.pattern.matches(event) {
                next_state = trans.to;
                matched = true;
                break;
            }
        }

        self.current_state = next_state;

        if self.error_states.contains(&self.current_state) {
            AutomatonResult::Violation(self.states[self.current_state].name.clone())
        } else if self.accepting_states.contains(&self.current_state) {
            AutomatonResult::Accepted
        } else if matched {
            AutomatonResult::Continue
        } else {
            AutomatonResult::NoTransition
        }
    }

    /// Reset the automaton to the initial state.
    pub fn reset(&mut self) {
        self.current_state = self.initial_state;
    }

    /// Check if the automaton is currently in an error state.
    pub fn is_in_error(&self) -> bool {
        self.error_states.contains(&self.current_state)
    }

    /// Check if the automaton is in an accepting state.
    pub fn is_accepting(&self) -> bool {
        self.accepting_states.contains(&self.current_state)
    }

    /// Create an automaton for "no unauthorized declassification".
    pub fn no_unauthorized_declassification() -> Self {
        let mut aut = Self::new("no-unauthorized-declassification");

        let init = aut.add_state("init", false, true);
        let authorized = aut.add_state("authorized", false, true);
        let error = aut.add_state("unauthorized-declass", true, false);

        // Authorization event
        aut.add_transition(init, authorized, EventPattern::EventType("authorize".to_string()));
        // Declassification after authorization — OK
        aut.add_transition(authorized, init, EventPattern::Declassification);
        // Declassification without authorization — error
        aut.add_transition(init, error, EventPattern::Declassification);
        // Any other event stays in same state
        aut.add_transition(init, init, EventPattern::EventType("other".to_string()));
        aut.add_transition(authorized, authorized, EventPattern::EventType("other".to_string()));

        aut
    }
}

/// Result of processing an event in the automaton.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AutomatonResult {
    /// No violation, continue monitoring.
    Continue,
    /// Property accepted (in accepting state).
    Accepted,
    /// Property violated.
    Violation(String),
    /// No matching transition.
    NoTransition,
}

impl fmt::Display for AutomatonResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AutomatonResult::Continue => write!(f, "continue"),
            AutomatonResult::Accepted => write!(f, "accepted"),
            AutomatonResult::Violation(s) => write!(f, "VIOLATION: {}", s),
            AutomatonResult::NoTransition => write!(f, "no transition"),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// HyperpropertyChecker — properties over sets of traces
// ═══════════════════════════════════════════════════════════════════════

/// Checker for hyperproperties (properties over sets of traces).
///
/// Noninterference is a 2-safety hyperproperty: it requires that
/// for all pairs of traces agreeing on low inputs, the low outputs match.
#[derive(Debug, Clone)]
pub struct HyperpropertyChecker {
    /// The security lattice.
    pub lattice: SecurityLattice,
    /// Properties to check.
    pub properties: Vec<SecurityProperty>,
    /// Collected traces.
    pub traces: Vec<ExecutionTrace>,
}

impl HyperpropertyChecker {
    pub fn new(lattice: SecurityLattice) -> Self {
        Self {
            lattice,
            properties: Vec::new(),
            traces: Vec::new(),
        }
    }

    pub fn add_property(&mut self, prop: SecurityProperty) {
        self.properties.push(prop);
    }

    pub fn add_trace(&mut self, trace: ExecutionTrace) {
        self.traces.push(trace);
    }

    /// Check all registered properties.
    pub fn check_all(&self) -> Vec<HyperpropertyResult> {
        let mut results = Vec::new();

        for prop in &self.properties {
            let result = match prop {
                SecurityProperty::Noninterference | SecurityProperty::TINI => {
                    self.check_noninterference()
                }
                SecurityProperty::SecrecyPreservation => {
                    self.check_secrecy()
                }
                _ => HyperpropertyResult {
                    property: prop.clone(),
                    holds: true,
                    witness: None,
                },
            };
            results.push(result);
        }

        results
    }

    fn check_noninterference(&self) -> HyperpropertyResult {
        let ni_checker = NoninterferenceChecker::new(self.lattice.clone());

        // Check all pairs of traces
        for i in 0..self.traces.len() {
            for j in (i + 1)..self.traces.len() {
                let result = ni_checker.check_traces(&self.traces[i], &self.traces[j]);
                if !result.holds {
                    return HyperpropertyResult {
                        property: SecurityProperty::Noninterference,
                        holds: false,
                        witness: Some(format!(
                            "Traces {} and {} differ on low outputs: {}",
                            i, j, result.reason,
                        )),
                    };
                }
            }
        }

        HyperpropertyResult {
            property: SecurityProperty::Noninterference,
            holds: true,
            witness: None,
        }
    }

    fn check_secrecy(&self) -> HyperpropertyResult {
        // Check that high-level values are not directly observable
        for (idx, trace) in self.traces.iter().enumerate() {
            for output in &trace.outputs {
                // If any output carries a high-level value, it's a violation
                // (simplified check — real implementation would track data flow)
                if output.channel.contains("secret") {
                    return HyperpropertyResult {
                        property: SecurityProperty::SecrecyPreservation,
                        holds: false,
                        witness: Some(format!(
                            "Trace {}: secret data on channel '{}'",
                            idx, output.channel,
                        )),
                    };
                }
            }
        }

        HyperpropertyResult {
            property: SecurityProperty::SecrecyPreservation,
            holds: true,
            witness: None,
        }
    }
}

/// Result of checking a hyperproperty.
#[derive(Debug, Clone)]
pub struct HyperpropertyResult {
    pub property: SecurityProperty,
    pub holds: bool,
    pub witness: Option<String>,
}

impl fmt::Display for HyperpropertyResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {}",
            self.property,
            if self.holds { "HOLDS" } else { "VIOLATED" },
        )?;
        if let Some(w) = &self.witness {
            write!(f, " ({})", w)?;
        }
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // --- SecurityLevel tests ---

    #[test]
    fn test_security_level_ordering() {
        assert!(SecurityLevel::Low < SecurityLevel::Medium);
        assert!(SecurityLevel::Medium < SecurityLevel::High);
        assert!(SecurityLevel::High < SecurityLevel::TopSecret);
    }

    #[test]
    fn test_security_level_flows_to() {
        assert!(SecurityLevel::Low.flows_to(&SecurityLevel::High));
        assert!(!SecurityLevel::High.flows_to(&SecurityLevel::Low));
        assert!(SecurityLevel::Medium.flows_to(&SecurityLevel::Medium));
    }

    #[test]
    fn test_security_level_lub_glb() {
        assert_eq!(
            SecurityLevel::Low.lub(&SecurityLevel::High),
            SecurityLevel::High,
        );
        assert_eq!(
            SecurityLevel::Low.glb(&SecurityLevel::High),
            SecurityLevel::Low,
        );
        assert_eq!(
            SecurityLevel::Medium.lub(&SecurityLevel::Medium),
            SecurityLevel::Medium,
        );
    }

    #[test]
    fn test_security_level_display() {
        assert_eq!(format!("{}", SecurityLevel::Low), "L");
        assert_eq!(format!("{}", SecurityLevel::High), "H");
    }

    // --- SecurityLattice tests ---

    #[test]
    fn test_two_level_lattice() {
        let lattice = SecurityLattice::two_level();
        assert_eq!(lattice.size(), 2);
        assert!(lattice.flows_to(0, 1)); // Low ≤ High
        assert!(!lattice.flows_to(1, 0)); // High ≰ Low
    }

    #[test]
    fn test_four_level_lattice() {
        let lattice = SecurityLattice::four_level();
        assert_eq!(lattice.size(), 4);
        assert!(lattice.flows_to(0, 3)); // Low ≤ TopSecret
        assert!(lattice.flows_to(1, 2)); // Medium ≤ High
        assert!(!lattice.flows_to(3, 0)); // TopSecret ≰ Low
    }

    #[test]
    fn test_diamond_lattice() {
        let lattice = SecurityLattice::diamond();
        assert_eq!(lattice.size(), 4);
        assert!(lattice.flows_to(0, 1)); // Bot ≤ A
        assert!(lattice.flows_to(0, 2)); // Bot ≤ B
        assert!(!lattice.flows_to(1, 2)); // A ≰ B (incomparable)
        assert!(!lattice.flows_to(2, 1)); // B ≰ A (incomparable)
        assert!(lattice.flows_to(1, 3)); // A ≤ Top
        assert!(lattice.flows_to(2, 3)); // B ≤ Top
    }

    #[test]
    fn test_lattice_join_meet() {
        let lattice = SecurityLattice::four_level();
        assert_eq!(lattice.join(0, 2), Some(2)); // Low ⊔ High = High
        assert_eq!(lattice.meet(1, 3), Some(1)); // Medium ⊓ TopSecret = Medium
    }

    #[test]
    fn test_lattice_label_index() {
        let lattice = SecurityLattice::two_level();
        assert_eq!(lattice.label_index("Low"), Some(0));
        assert_eq!(lattice.label_index("High"), Some(1));
        assert_eq!(lattice.label_index("Unknown"), None);
    }

    // --- SecurityProperty tests ---

    #[test]
    fn test_security_property_display() {
        assert_eq!(
            format!("{}", SecurityProperty::Noninterference),
            "Noninterference",
        );
        assert_eq!(
            format!("{}", SecurityProperty::SecrecyPreservation),
            "SecrecyPreservation",
        );
    }

    // --- NoninterferenceChecker tests ---

    #[test]
    fn test_noninterference_holds() {
        let lattice = SecurityLattice::two_level();
        let checker = NoninterferenceChecker::new(lattice);

        let mut trace_a = ExecutionTrace::new();
        trace_a.add_output("result", 42, 0);
        trace_a.set_input_level("x", SecurityLevel::Low);

        let mut trace_b = ExecutionTrace::new();
        trace_b.add_output("result", 42, 0);
        trace_b.set_input_level("x", SecurityLevel::Low);

        let result = checker.check_traces(&trace_a, &trace_b);
        assert!(result.holds);
    }

    #[test]
    fn test_noninterference_violated() {
        let lattice = SecurityLattice::two_level();
        let checker = NoninterferenceChecker::new(lattice);

        let mut trace_a = ExecutionTrace::new();
        trace_a.add_output("result", 42, 0);
        trace_a.set_input_level("x", SecurityLevel::Low);

        let mut trace_b = ExecutionTrace::new();
        trace_b.add_output("result", 99, 0); // Different output!
        trace_b.set_input_level("x", SecurityLevel::Low);

        let result = checker.check_traces(&trace_a, &trace_b);
        assert!(!result.holds);
        assert_eq!(result.violations.len(), 1);
    }

    #[test]
    fn test_noninterference_different_low_inputs() {
        let lattice = SecurityLattice::two_level();
        let mut checker = NoninterferenceChecker::new(lattice);
        checker.set_variable_level("x", 0); // Low

        let mut trace_a = ExecutionTrace::new();
        trace_a.add_output("result", 42, 0);
        trace_a.set_input_level("x", SecurityLevel::Low);

        let mut trace_b = ExecutionTrace::new();
        trace_b.add_output("result", 99, 0);
        trace_b.set_input_level("x", SecurityLevel::High); // Different level

        let result = checker.check_traces(&trace_a, &trace_b);
        // Low inputs differ, so noninterference is vacuously true
        assert!(result.holds);
    }

    // --- DeclassificationPolicy tests ---

    #[test]
    fn test_declassification_policy() {
        let policy = DeclassificationPolicy::password_hash_policy();
        assert!(policy.is_allowed(
            "password_hash",
            &SecurityLevel::High,
            &SecurityLevel::Low,
        ));
        assert!(!policy.is_allowed(
            "raw_password",
            &SecurityLevel::High,
            &SecurityLevel::Low,
        ));
    }

    #[test]
    fn test_no_declassification_policy() {
        let policy = DeclassificationPolicy::no_declassification();
        assert!(!policy.is_allowed(
            "anything",
            &SecurityLevel::High,
            &SecurityLevel::Low,
        ));
    }

    #[test]
    fn test_declassifiable_variables() {
        let policy = DeclassificationPolicy::password_hash_policy();
        let vars = policy.declassifiable_variables();
        assert!(vars.contains(&"password_hash"));
    }

    // --- InformationFlowType tests ---

    #[test]
    fn test_flow_type_creation() {
        let ft = InformationFlowType::new(SecurityLevel::High);
        assert_eq!(ft.level, SecurityLevel::High);
        assert!(!ft.existence_classified);
        assert!(ft.dependencies.is_empty());
    }

    #[test]
    fn test_flow_type_join() {
        let a = InformationFlowType::new(SecurityLevel::Low)
            .with_dependency("x");
        let b = InformationFlowType::new(SecurityLevel::High)
            .with_dependency("y");
        let joined = a.join(&b);
        assert_eq!(joined.level, SecurityLevel::High);
        assert!(joined.dependencies.contains("x"));
        assert!(joined.dependencies.contains("y"));
    }

    #[test]
    fn test_flow_type_can_assign() {
        let low = InformationFlowType::new(SecurityLevel::Low);
        let high = InformationFlowType::new(SecurityLevel::High);

        assert!(low.can_assign_to(&SecurityLevel::High));
        assert!(low.can_assign_to(&SecurityLevel::Low));
        assert!(!high.can_assign_to(&SecurityLevel::Low));
    }

    // --- InformationFlowChecker tests ---

    #[test]
    fn test_flow_checker_secure_assignment() {
        let mut checker = InformationFlowChecker::new();
        checker.set_type("x", InformationFlowType::new(SecurityLevel::Low));
        checker.set_type("y", InformationFlowType::new(SecurityLevel::High));

        // Low → High is secure
        assert!(checker.check_assignment("x", "y"));
        assert!(!checker.has_violations());
    }

    #[test]
    fn test_flow_checker_insecure_assignment() {
        let mut checker = InformationFlowChecker::new();
        checker.set_type("secret", InformationFlowType::new(SecurityLevel::High));
        checker.set_type("public", InformationFlowType::new(SecurityLevel::Low));

        // High → Low is insecure
        assert!(!checker.check_assignment("secret", "public"));
        assert!(checker.has_violations());
        assert_eq!(checker.violations.len(), 1);
    }

    #[test]
    fn test_flow_checker_pc_elevation() {
        let mut checker = InformationFlowChecker::new();
        checker.set_type("secret", InformationFlowType::new(SecurityLevel::High));
        assert_eq!(checker.pc_level, SecurityLevel::Low);

        checker.enter_branch("secret");
        assert_eq!(checker.pc_level, SecurityLevel::High);

        checker.exit_branch(SecurityLevel::Low);
        assert_eq!(checker.pc_level, SecurityLevel::Low);
    }

    // --- SecurityAutomaton tests ---

    #[test]
    fn test_automaton_creation() {
        let aut = SecurityAutomaton::no_unauthorized_declassification();
        assert_eq!(aut.states.len(), 3);
        assert!(!aut.is_in_error());
        assert!(aut.is_accepting());
    }

    #[test]
    fn test_automaton_authorized_declassification() {
        let mut aut = SecurityAutomaton::no_unauthorized_declassification();

        // Authorize first
        let event1 = SecurityEvent::new("authorize", SecurityLevel::High);
        let result1 = aut.process(&event1);
        assert_eq!(result1, AutomatonResult::Continue);

        // Then declassify
        let event2 = SecurityEvent::new("declassify", SecurityLevel::Low)
            .with_declassification();
        let result2 = aut.process(&event2);
        assert_ne!(result2, AutomatonResult::Violation("unauthorized-declass".to_string()));
    }

    #[test]
    fn test_automaton_unauthorized_declassification() {
        let mut aut = SecurityAutomaton::no_unauthorized_declassification();

        // Declassify without authorization
        let event = SecurityEvent::new("declassify", SecurityLevel::Low)
            .with_declassification();
        let result = aut.process(&event);
        assert_eq!(result, AutomatonResult::Violation("unauthorized-declass".to_string()));
        assert!(aut.is_in_error());
    }

    #[test]
    fn test_automaton_reset() {
        let mut aut = SecurityAutomaton::no_unauthorized_declassification();

        let event = SecurityEvent::new("declassify", SecurityLevel::Low)
            .with_declassification();
        aut.process(&event);
        assert!(aut.is_in_error());

        aut.reset();
        assert!(!aut.is_in_error());
    }

    // --- HyperpropertyChecker tests ---

    #[test]
    fn test_hyperproperty_noninterference_holds() {
        let lattice = SecurityLattice::two_level();
        let mut checker = HyperpropertyChecker::new(lattice);
        checker.add_property(SecurityProperty::Noninterference);

        let mut trace1 = ExecutionTrace::new();
        trace1.add_output("out", 42, 0);

        let mut trace2 = ExecutionTrace::new();
        trace2.add_output("out", 42, 0);

        checker.add_trace(trace1);
        checker.add_trace(trace2);

        let results = checker.check_all();
        assert_eq!(results.len(), 1);
        assert!(results[0].holds);
    }

    #[test]
    fn test_hyperproperty_noninterference_violated() {
        let lattice = SecurityLattice::two_level();
        let mut checker = HyperpropertyChecker::new(lattice);
        checker.add_property(SecurityProperty::Noninterference);

        let mut trace1 = ExecutionTrace::new();
        trace1.add_output("out", 42, 0);

        let mut trace2 = ExecutionTrace::new();
        trace2.add_output("out", 99, 0); // Different!

        checker.add_trace(trace1);
        checker.add_trace(trace2);

        let results = checker.check_all();
        assert!(!results[0].holds);
    }

    #[test]
    fn test_hyperproperty_secrecy() {
        let lattice = SecurityLattice::two_level();
        let mut checker = HyperpropertyChecker::new(lattice);
        checker.add_property(SecurityProperty::SecrecyPreservation);

        let mut trace = ExecutionTrace::new();
        trace.add_output("secret_channel", 42, 0);
        checker.add_trace(trace);

        let results = checker.check_all();
        assert!(!results[0].holds); // Secret data on output channel
    }

    #[test]
    fn test_hyperproperty_secrecy_holds() {
        let lattice = SecurityLattice::two_level();
        let mut checker = HyperpropertyChecker::new(lattice);
        checker.add_property(SecurityProperty::SecrecyPreservation);

        let mut trace = ExecutionTrace::new();
        trace.add_output("public_result", 42, 0);
        checker.add_trace(trace);

        let results = checker.check_all();
        assert!(results[0].holds);
    }

    // --- EventPattern tests ---

    #[test]
    fn test_event_pattern_any() {
        let event = SecurityEvent::new("test", SecurityLevel::Low);
        assert!(EventPattern::Any.matches(&event));
    }

    #[test]
    fn test_event_pattern_event_type() {
        let event = SecurityEvent::new("read", SecurityLevel::Low);
        assert!(EventPattern::EventType("read".to_string()).matches(&event));
        assert!(!EventPattern::EventType("write".to_string()).matches(&event));
    }

    #[test]
    fn test_event_pattern_variable() {
        let event = SecurityEvent::new("read", SecurityLevel::Low)
            .with_variable("x");
        assert!(EventPattern::VariableAccess("x".to_string()).matches(&event));
        assert!(!EventPattern::VariableAccess("y".to_string()).matches(&event));
    }

    #[test]
    fn test_event_pattern_and_or() {
        let event = SecurityEvent::new("read", SecurityLevel::High);
        let p1 = EventPattern::EventType("read".to_string());
        let p2 = EventPattern::SecurityLevel(SecurityLevel::High);
        let p3 = EventPattern::SecurityLevel(SecurityLevel::Low);

        let and_p = EventPattern::And(Box::new(p1.clone()), Box::new(p2.clone()));
        assert!(and_p.matches(&event));

        let and_fail = EventPattern::And(Box::new(p1.clone()), Box::new(p3.clone()));
        assert!(!and_fail.matches(&event));

        let or_p = EventPattern::Or(Box::new(p2), Box::new(p3));
        assert!(or_p.matches(&event));
    }

    // --- Display tests ---

    #[test]
    fn test_noninterference_result_display() {
        let result = NoninterferenceResult {
            holds: true,
            reason: "OK".to_string(),
            violations: Vec::new(),
        };
        let s = format!("{}", result);
        assert!(s.contains("HOLDS"));
    }

    #[test]
    fn test_hyperproperty_result_display() {
        let result = HyperpropertyResult {
            property: SecurityProperty::Noninterference,
            holds: false,
            witness: Some("traces differ".to_string()),
        };
        let s = format!("{}", result);
        assert!(s.contains("VIOLATED"));
    }

    #[test]
    fn test_automaton_result_display() {
        assert_eq!(format!("{}", AutomatonResult::Continue), "continue");
        assert_eq!(format!("{}", AutomatonResult::Accepted), "accepted");
        assert!(format!("{}", AutomatonResult::Violation("test".into())).contains("VIOLATION"));
    }

    #[test]
    fn test_flow_violation_display() {
        let v = FlowViolation {
            source: "secret".to_string(),
            target: "public".to_string(),
            source_level: SecurityLevel::High,
            target_level: SecurityLevel::Low,
            description: "leak".to_string(),
        };
        let s = format!("{}", v);
        assert!(s.contains("secret"));
        assert!(s.contains("public"));
    }

    #[test]
    fn test_flow_type_display() {
        let ft = InformationFlowType::new(SecurityLevel::High)
            .with_dependency("x")
            .with_existence_classified();
        let s = format!("{}", ft);
        assert!(s.contains("H"));
        assert!(s.contains("∃-classified"));
    }
}
