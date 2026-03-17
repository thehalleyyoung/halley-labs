//! Symbolic trace management for the extraction pipeline.
//!
//! Provides types and algorithms for collecting, normalizing, merging, and
//! filtering symbolic execution traces before they are consumed by the
//! state machine extractor.

use crate::{
    BinOp, ExtractError, ExtractResult, HandshakePhase, MessageLabel, NegotiationState, Observable,
    ProtocolVersion, StateId, SymbolicState, SymbolicValue,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt;

// ---------------------------------------------------------------------------
// Core trace types
// ---------------------------------------------------------------------------

/// The type of action recorded in a trace step.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TraceActionType {
    /// A protocol message was sent or received.
    ProtocolMessage,
    /// An internal computation step.
    InternalComputation,
    /// A branch decision in the program.
    BranchDecision,
    /// An adversary action.
    AdversaryAction,
    /// A merge point where states were combined.
    MergePoint,
}

/// A single step in a symbolic execution trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolicTraceStep {
    /// The symbolic state at this point.
    pub state: SymbolicState,
    /// The label of the action taken from this state.
    pub label: MessageLabel,
    /// Classification of this step.
    pub action_type: TraceActionType,
    /// Path constraint added at this step (if any).
    pub added_constraint: Option<SymbolicValue>,
    /// Whether this step is relevant to negotiation.
    pub is_negotiation_relevant: bool,
    /// Monotonic step index within the trace.
    pub step_index: u32,
}

impl SymbolicTraceStep {
    pub fn new(state: SymbolicState, label: MessageLabel, action_type: TraceActionType) -> Self {
        let is_relevant = matches!(
            action_type,
            TraceActionType::ProtocolMessage | TraceActionType::AdversaryAction
        );
        Self {
            state,
            label,
            action_type,
            added_constraint: None,
            is_negotiation_relevant: is_relevant,
            step_index: 0,
        }
    }

    pub fn with_constraint(mut self, constraint: SymbolicValue) -> Self {
        self.added_constraint = Some(constraint);
        self
    }

    pub fn phase(&self) -> HandshakePhase {
        self.state.negotiation.phase
    }
}

/// A complete symbolic execution trace: a sequence of (state, action, constraint) triples.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolicTrace {
    /// Ordered sequence of trace steps.
    pub steps: Vec<SymbolicTraceStep>,
    /// Unique identifier for this trace.
    pub trace_id: u64,
    /// Whether this trace reached a terminal state.
    pub is_complete: bool,
    /// The final outcome (for complete traces).
    pub final_outcome: Option<Observable>,
    /// Set of symbolic state IDs visited along this trace.
    pub visited_state_ids: Vec<u64>,
}

impl SymbolicTrace {
    pub fn new(trace_id: u64) -> Self {
        Self {
            steps: Vec::new(),
            trace_id,
            is_complete: false,
            final_outcome: None,
            visited_state_ids: Vec::new(),
        }
    }

    pub fn push_step(&mut self, mut step: SymbolicTraceStep) {
        step.step_index = self.steps.len() as u32;
        self.visited_state_ids.push(step.state.id);
        self.steps.push(step);
    }

    pub fn len(&self) -> usize {
        self.steps.len()
    }

    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Returns the initial state of the trace (if any).
    pub fn initial_state(&self) -> Option<&SymbolicState> {
        self.steps.first().map(|s| &s.state)
    }

    /// Returns the final state of the trace (if any).
    pub fn final_state(&self) -> Option<&SymbolicState> {
        self.steps.last().map(|s| &s.state)
    }

    /// The sequence of message labels along this trace.
    pub fn label_sequence(&self) -> Vec<&MessageLabel> {
        self.steps.iter().map(|s| &s.label).collect()
    }

    /// All phases visited by this trace.
    pub fn phases_visited(&self) -> Vec<HandshakePhase> {
        let mut phases = Vec::new();
        let mut seen = HashSet::new();
        for step in &self.steps {
            let phase = step.state.negotiation.phase;
            if seen.insert(phase) {
                phases.push(phase);
            }
        }
        phases
    }

    /// Aggregate all path constraints accumulated along the trace.
    pub fn aggregate_constraints(&self) -> Vec<SymbolicValue> {
        let mut constraints = Vec::new();
        for step in &self.steps {
            for pc in &step.state.constraints {
                constraints.push(pc.clone());
            }
            if let Some(ref c) = step.added_constraint {
                constraints.push(c.clone());
            }
        }
        constraints
    }

    /// Branching factor: how many unique successors from each step.
    pub fn max_branching_factor(&self, all_traces: &[SymbolicTrace]) -> usize {
        let mut max_bf = 0;
        for step in &self.steps {
            let sid = step.state.id;
            let mut successors = HashSet::new();
            for trace in all_traces {
                for window in trace.steps.windows(2) {
                    if window[0].state.id == sid {
                        successors.insert(window[1].state.id);
                    }
                }
            }
            max_bf = max_bf.max(successors.len());
        }
        max_bf
    }

    /// Mark the trace complete with a final observation.
    pub fn mark_complete(&mut self, outcome: Observable) {
        self.is_complete = true;
        self.final_outcome = Some(outcome);
    }

    /// Compute a canonical fingerprint of this trace for dedup purposes.
    /// Uses the sequence of (phase, label_name) pairs.
    pub fn fingerprint(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        for step in &self.steps {
            step.state.negotiation.phase.hash(&mut hasher);
            step.label.label_name().hash(&mut hasher);
            if let Some(ref c) = step.state.negotiation.selected_cipher {
                c.iana_id.hash(&mut hasher);
            }
        }
        hasher.finish()
    }
}

impl fmt::Display for SymbolicTrace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Trace#{} [{}steps, complete={}]",
            self.trace_id,
            self.steps.len(),
            self.is_complete
        )?;
        for (i, step) in self.steps.iter().enumerate() {
            write!(
                f,
                "\n  {}: {} --[{}]--> ...",
                i,
                step.state.negotiation.phase,
                step.label.label_name()
            )?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// TraceCollector
// ---------------------------------------------------------------------------

/// Gathers traces from symbolic execution, assigning unique IDs and
/// tracking which states have been visited across all traces.
pub struct TraceCollector {
    traces: Vec<SymbolicTrace>,
    next_trace_id: u64,
    seen_state_ids: HashSet<u64>,
    max_trace_count: usize,
}

impl TraceCollector {
    pub fn new(max_trace_count: usize) -> Self {
        Self {
            traces: Vec::new(),
            next_trace_id: 0,
            seen_state_ids: HashSet::new(),
            max_trace_count,
        }
    }

    /// Begin a new trace. Returns its ID.
    pub fn begin_trace(&mut self) -> u64 {
        let id = self.next_trace_id;
        self.next_trace_id += 1;
        let trace = SymbolicTrace::new(id);
        self.traces.push(trace);
        id
    }

    /// Append a step to the current (latest) trace.
    pub fn record_step(
        &mut self,
        state: SymbolicState,
        label: MessageLabel,
        action_type: TraceActionType,
    ) -> ExtractResult<()> {
        let trace = self
            .traces
            .last_mut()
            .ok_or(ExtractError::Internal("no active trace".into()))?;
        self.seen_state_ids.insert(state.id);
        let step = SymbolicTraceStep::new(state, label, action_type);
        trace.push_step(step);
        Ok(())
    }

    /// Append a step with an associated constraint.
    pub fn record_step_with_constraint(
        &mut self,
        state: SymbolicState,
        label: MessageLabel,
        action_type: TraceActionType,
        constraint: SymbolicValue,
    ) -> ExtractResult<()> {
        let trace = self
            .traces
            .last_mut()
            .ok_or(ExtractError::Internal("no active trace".into()))?;
        self.seen_state_ids.insert(state.id);
        let step = SymbolicTraceStep::new(state, label, action_type).with_constraint(constraint);
        trace.push_step(step);
        Ok(())
    }

    /// Mark the current trace as complete.
    pub fn complete_trace(&mut self, outcome: Observable) -> ExtractResult<()> {
        let trace = self
            .traces
            .last_mut()
            .ok_or(ExtractError::Internal("no active trace".into()))?;
        trace.mark_complete(outcome);
        Ok(())
    }

    /// Add an externally-constructed trace.
    pub fn add_trace(&mut self, mut trace: SymbolicTrace) {
        trace.trace_id = self.next_trace_id;
        self.next_trace_id += 1;
        for step in &trace.steps {
            self.seen_state_ids.insert(step.state.id);
        }
        self.traces.push(trace);
    }

    /// Consume the collector and return all traces.
    pub fn into_traces(self) -> Vec<SymbolicTrace> {
        self.traces
    }

    /// Borrow the collected traces.
    pub fn traces(&self) -> &[SymbolicTrace] {
        &self.traces
    }

    pub fn trace_count(&self) -> usize {
        self.traces.len()
    }

    pub fn unique_states_seen(&self) -> usize {
        self.seen_state_ids.len()
    }

    pub fn at_capacity(&self) -> bool {
        self.traces.len() >= self.max_trace_count
    }
}

// ---------------------------------------------------------------------------
// TraceNormalizer
// ---------------------------------------------------------------------------

/// Canonicalizes traces for comparison by:
/// - Re-indexing step indices from 0
/// - Sorting constraints within each step
/// - Collapsing consecutive Tau steps
/// - Normalizing state IDs to a canonical ordering
pub struct TraceNormalizer;

impl TraceNormalizer {
    /// Normalize a single trace.
    pub fn normalize(trace: &SymbolicTrace) -> SymbolicTrace {
        let mut normalized = SymbolicTrace::new(trace.trace_id);
        normalized.is_complete = trace.is_complete;
        normalized.final_outcome = trace.final_outcome.clone();

        let mut prev_was_tau = false;
        let mut step_idx = 0u32;

        for step in &trace.steps {
            // Collapse consecutive Tau steps: keep the last one.
            if step.label == MessageLabel::Tau {
                if prev_was_tau {
                    // Remove the previously added Tau step and replace
                    if let Some(last) = normalized.steps.last_mut() {
                        last.state = step.state.clone();
                        last.added_constraint = step.added_constraint.clone();
                    }
                    continue;
                }
                prev_was_tau = true;
            } else {
                prev_was_tau = false;
            }

            let mut new_step = step.clone();
            new_step.step_index = step_idx;
            step_idx += 1;

            // Sort the path constraints for canonical form.
            new_step.state.constraints.sort_by(|a, b| {
                format!("{}", a).cmp(&format!("{}", b))
            });

            normalized.push_step(new_step);
        }
        normalized
    }

    /// Normalize all traces in a batch.
    pub fn normalize_all(traces: &[SymbolicTrace]) -> Vec<SymbolicTrace> {
        traces.iter().map(|t| Self::normalize(t)).collect()
    }

    /// Assign canonical state IDs based on first-visit order across all traces.
    pub fn canonicalize_state_ids(traces: &mut [SymbolicTrace]) {
        let mut id_map: HashMap<u64, u64> = HashMap::new();
        let mut next_canonical = 0u64;

        for trace in traces.iter() {
            for step in &trace.steps {
                let old_id = step.state.id;
                if !id_map.contains_key(&old_id) {
                    id_map.insert(old_id, next_canonical);
                    next_canonical += 1;
                }
            }
        }

        for trace in traces.iter_mut() {
            for step in &mut trace.steps {
                if let Some(&new_id) = id_map.get(&step.state.id) {
                    step.state.id = new_id;
                    step.state.parent_id = step
                        .state
                        .parent_id
                        .and_then(|pid| id_map.get(&pid).copied());
                }
            }
            trace.visited_state_ids = trace
                .visited_state_ids
                .iter()
                .filter_map(|id| id_map.get(id).copied())
                .collect();
        }
    }
}

// ---------------------------------------------------------------------------
// TraceMerger
// ---------------------------------------------------------------------------

/// Combines traces that visit the same sequence of states, merging
/// their constraints to produce a single representative trace.
pub struct TraceMerger;

impl TraceMerger {
    /// Group traces by their fingerprint and merge each group.
    pub fn merge_traces(traces: Vec<SymbolicTrace>) -> Vec<SymbolicTrace> {
        let mut groups: BTreeMap<u64, Vec<SymbolicTrace>> = BTreeMap::new();
        for trace in traces {
            let fp = trace.fingerprint();
            groups.entry(fp).or_default().push(trace);
        }

        let mut merged = Vec::new();
        for (_fp, group) in groups {
            if group.len() == 1 {
                merged.push(group.into_iter().next().unwrap());
            } else {
                merged.push(Self::merge_group(group));
            }
        }
        merged
    }

    /// Merge a group of traces with the same fingerprint.
    ///
    /// Takes the first trace as a template, and for each step, builds
    /// ITE expressions over the distinct constraints from all traces
    /// in the group.
    fn merge_group(group: Vec<SymbolicTrace>) -> SymbolicTrace {
        assert!(!group.is_empty());
        let first = &group[0];
        let mut result = SymbolicTrace::new(first.trace_id);
        result.is_complete = first.is_complete;
        result.final_outcome = first.final_outcome.clone();

        let min_len = group.iter().map(|t| t.steps.len()).min().unwrap_or(0);
        for step_idx in 0..min_len {
            let mut merged_step = group[0].steps[step_idx].clone();

            // Collect unique added constraints from all traces at this step.
            let mut constraint_set: Vec<SymbolicValue> = Vec::new();
            let mut seen_strs: HashSet<String> = HashSet::new();
            for trace in &group {
                if step_idx < trace.steps.len() {
                    if let Some(ref c) = trace.steps[step_idx].added_constraint {
                        let s = format!("{}", c);
                        if seen_strs.insert(s) {
                            constraint_set.push(c.clone());
                        }
                    }
                }
            }

            // If there are multiple distinct constraints, combine with OR.
            if constraint_set.len() > 1 {
                let combined = constraint_set.into_iter().reduce(|acc, c| {
                    SymbolicValue::binary(BinOp::LogicOr, acc, c)
                });
                merged_step.added_constraint = combined;
            } else if constraint_set.len() == 1 {
                merged_step.added_constraint = Some(constraint_set.remove(0));
            }

            merged_step.action_type = TraceActionType::MergePoint;
            result.push_step(merged_step);
        }
        result
    }
}

// ---------------------------------------------------------------------------
// TraceFilter
// ---------------------------------------------------------------------------

/// Removes trace steps that are not relevant to negotiation.
pub struct TraceFilter;

impl TraceFilter {
    /// Filter a single trace, keeping only negotiation-relevant steps.
    pub fn filter(trace: &SymbolicTrace) -> SymbolicTrace {
        let mut filtered = SymbolicTrace::new(trace.trace_id);
        filtered.is_complete = trace.is_complete;
        filtered.final_outcome = trace.final_outcome.clone();

        for step in &trace.steps {
            if Self::is_relevant(step) {
                filtered.push_step(step.clone());
            }
        }
        filtered
    }

    /// Filter all traces.
    pub fn filter_all(traces: &[SymbolicTrace]) -> Vec<SymbolicTrace> {
        traces
            .iter()
            .map(|t| Self::filter(t))
            .filter(|t| !t.is_empty())
            .collect()
    }

    /// Determine if a step is relevant to the negotiation protocol.
    fn is_relevant(step: &SymbolicTraceStep) -> bool {
        if step.is_negotiation_relevant {
            return true;
        }

        // Keep branch decisions that involve cipher/version selection.
        if step.action_type == TraceActionType::BranchDecision {
            if let Some(ref constraint) = step.added_constraint {
                return Self::involves_negotiation_variable(constraint);
            }
        }

        // Keep merge points.
        if step.action_type == TraceActionType::MergePoint {
            return true;
        }

        // Keep any step where the negotiation state changes.
        // (Caller can detect this by comparing adjacent steps.)
        matches!(
            step.state.negotiation.phase,
            HandshakePhase::ClientHello
                | HandshakePhase::ServerHello
                | HandshakePhase::Certificate
                | HandshakePhase::KeyExchange
                | HandshakePhase::ChangeCipherSpec
                | HandshakePhase::Finished
                | HandshakePhase::Alert
        )
    }

    /// Heuristic: does this symbolic value reference negotiation-related variables?
    fn involves_negotiation_variable(val: &SymbolicValue) -> bool {
        match val {
            SymbolicValue::Variable(sym_var) => {
                let n = sym_var.name.to_lowercase();
                n.contains("cipher")
                    || n.contains("suite")
                    || n.contains("version")
                    || n.contains("extension")
                    || n.contains("tls")
                    || n.contains("ssl")
                    || n.contains("negotiate")
                    || n.contains("handshake")
            }
            SymbolicValue::UnaryOp { operand: a, .. } => {
                Self::involves_negotiation_variable(a)
            }
            SymbolicValue::BinaryOp { left: a, right: b, .. } => {
                Self::involves_negotiation_variable(a)
                    || Self::involves_negotiation_variable(b)
            }
            SymbolicValue::Select { array: a, index: b } => {
                Self::involves_negotiation_variable(a)
                    || Self::involves_negotiation_variable(b)
            }
            SymbolicValue::Ite { condition: c, then_val: t, else_val: e }
            | SymbolicValue::Store { array: c, index: t, value: e } => {
                Self::involves_negotiation_variable(c)
                    || Self::involves_negotiation_variable(t)
                    || Self::involves_negotiation_variable(e)
            }
            _ => false,
        }
    }
}

// ---------------------------------------------------------------------------
// TraceStatistics
// ---------------------------------------------------------------------------

/// Statistics over a collection of traces.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TraceStatistics {
    pub total_traces: usize,
    pub complete_traces: usize,
    pub incomplete_traces: usize,
    pub min_length: usize,
    pub max_length: usize,
    pub mean_length: f64,
    pub median_length: usize,
    pub total_steps: usize,
    pub unique_states: usize,
    pub unique_labels: usize,
    pub max_branching_factor: usize,
    pub protocol_message_steps: usize,
    pub adversary_steps: usize,
    pub internal_steps: usize,
    pub merge_point_steps: usize,
    pub length_histogram: BTreeMap<usize, usize>,
}

impl TraceStatistics {
    pub fn compute(traces: &[SymbolicTrace]) -> Self {
        if traces.is_empty() {
            return Self::default();
        }

        let mut stats = Self {
            total_traces: traces.len(),
            ..Default::default()
        };

        let mut lengths: Vec<usize> = Vec::new();
        let mut state_ids: HashSet<u64> = HashSet::new();
        let mut label_names: HashSet<String> = HashSet::new();

        for trace in traces {
            let len = trace.len();
            lengths.push(len);
            stats.total_steps += len;

            if trace.is_complete {
                stats.complete_traces += 1;
            } else {
                stats.incomplete_traces += 1;
            }

            *stats.length_histogram.entry(len).or_insert(0) += 1;

            for step in &trace.steps {
                state_ids.insert(step.state.id);
                label_names.insert(step.label.label_name().to_string());

                match step.action_type {
                    TraceActionType::ProtocolMessage => stats.protocol_message_steps += 1,
                    TraceActionType::AdversaryAction => stats.adversary_steps += 1,
                    TraceActionType::InternalComputation
                    | TraceActionType::BranchDecision => stats.internal_steps += 1,
                    TraceActionType::MergePoint => stats.merge_point_steps += 1,
                }
            }
        }

        lengths.sort_unstable();
        stats.min_length = *lengths.first().unwrap_or(&0);
        stats.max_length = *lengths.last().unwrap_or(&0);
        stats.mean_length = if lengths.is_empty() {
            0.0
        } else {
            lengths.iter().sum::<usize>() as f64 / lengths.len() as f64
        };
        stats.median_length = if lengths.is_empty() {
            0
        } else {
            lengths[lengths.len() / 2]
        };

        stats.unique_states = state_ids.len();
        stats.unique_labels = label_names.len();

        // Compute max branching factor across all traces.
        let mut successor_map: HashMap<u64, HashSet<u64>> = HashMap::new();
        for trace in traces {
            for window in trace.steps.windows(2) {
                successor_map
                    .entry(window[0].state.id)
                    .or_default()
                    .insert(window[1].state.id);
            }
        }
        stats.max_branching_factor = successor_map
            .values()
            .map(|s| s.len())
            .max()
            .unwrap_or(0);

        stats
    }
}

impl fmt::Display for TraceStatistics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Trace Statistics:")?;
        writeln!(
            f,
            "  Total: {} ({} complete, {} incomplete)",
            self.total_traces, self.complete_traces, self.incomplete_traces
        )?;
        writeln!(
            f,
            "  Length: min={}, max={}, mean={:.1}, median={}",
            self.min_length, self.max_length, self.mean_length, self.median_length
        )?;
        writeln!(
            f,
            "  Steps: {} total ({} proto, {} adv, {} internal, {} merge)",
            self.total_steps,
            self.protocol_message_steps,
            self.adversary_steps,
            self.internal_steps,
            self.merge_point_steps
        )?;
        writeln!(
            f,
            "  Unique states: {}, unique labels: {}, max BF: {}",
            self.unique_states, self.unique_labels, self.max_branching_factor
        )
    }
}

// ---------------------------------------------------------------------------
// Path constraint aggregation
// ---------------------------------------------------------------------------

/// Aggregates and simplifies path constraints along traces.
pub struct ConstraintAggregator;

impl ConstraintAggregator {
    /// Collect all constraints from a trace into a single conjunction.
    pub fn aggregate(trace: &SymbolicTrace) -> Option<SymbolicValue> {
        let constraints = trace.aggregate_constraints();
        if constraints.is_empty() {
            return None;
        }
        Some(Self::conjoin(constraints))
    }

    /// Build the conjunction of a set of constraints.
    pub fn conjoin(constraints: Vec<SymbolicValue>) -> SymbolicValue {
        if constraints.is_empty() {
            return SymbolicValue::bool_const(true);
        }
        constraints
            .into_iter()
            .reduce(|acc, c| SymbolicValue::binary(BinOp::LogicAnd, acc, c))
            .unwrap()
    }

    /// Extract constraints specific to cipher selection from a trace.
    pub fn cipher_constraints(trace: &SymbolicTrace) -> Vec<SymbolicValue> {
        let mut result = Vec::new();
        for step in &trace.steps {
            if let Some(ref c) = step.added_constraint {
                if TraceFilter::involves_negotiation_variable(c) {
                    result.push(c.clone());
                }
            }
            for pc in &step.state.constraints {
                if TraceFilter::involves_negotiation_variable(pc) {
                    result.push(pc.clone());
                }
            }
        }
        result
    }

    /// Count the total number of constraint nodes across a trace.
    pub fn total_constraint_complexity(trace: &SymbolicTrace) -> usize {
        let mut total = 0;
        for step in &trace.steps {
            for pc in &step.state.constraints {
                total += pc.node_count();
            }
            if let Some(ref c) = step.added_constraint {
                total += c.node_count();
            }
        }
        total
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use negsyn_types::PathConstraint;
    use std::collections::BTreeSet;

    fn make_sym_state(id: u64, phase: HandshakePhase) -> SymbolicState {
        let mut ns = NegotiationState::new();
        ns.phase = phase;
        ns.version = Some(ProtocolVersion::Tls12);
        let mut s = SymbolicState::new(id, 0x1000 + id);
        s.negotiation = ns;
        s
    }

    fn make_step(
        id: u64,
        phase: HandshakePhase,
        label: MessageLabel,
        action: TraceActionType,
    ) -> SymbolicTraceStep {
        SymbolicTraceStep::new(make_sym_state(id, phase), label, action)
    }

    fn make_simple_trace() -> SymbolicTrace {
        let mut trace = SymbolicTrace::new(0);
        trace.push_step(make_step(
            0,
            HandshakePhase::Initial,
            MessageLabel::Tau,
            TraceActionType::InternalComputation,
        ));
        trace.push_step(make_step(
            1,
            HandshakePhase::ClientHello,
            MessageLabel::ClientHello {
                offered_ciphers: [0x002f, 0x0035].into(),
                version: ProtocolVersion::Tls12,
                extensions: vec![],
            },
            TraceActionType::ProtocolMessage,
        ));
        trace.push_step(make_step(
            2,
            HandshakePhase::ServerHello,
            MessageLabel::ServerHello {
                selected_cipher: 0x002f,
                version: ProtocolVersion::Tls12,
                extensions: vec![],
            },
            TraceActionType::ProtocolMessage,
        ));
        trace.mark_complete(Observable::successful(
            0x002f,
            ProtocolVersion::Tls12,
            BTreeSet::new(),
        ));
        trace
    }

    #[test]
    fn test_trace_basic() {
        let trace = make_simple_trace();
        assert_eq!(trace.len(), 3);
        assert!(trace.is_complete);
        assert_eq!(trace.visited_state_ids, vec![0, 1, 2]);
    }

    #[test]
    fn test_trace_label_sequence() {
        let trace = make_simple_trace();
        let labels: Vec<&str> = trace.label_sequence().iter().map(|l| l.label_name()).collect();
        assert_eq!(labels, vec!["τ", "CH", "SH"]);
    }

    #[test]
    fn test_trace_phases_visited() {
        let trace = make_simple_trace();
        let phases = trace.phases_visited();
        assert_eq!(
            phases,
            vec![
                HandshakePhase::Initial,
                HandshakePhase::ClientHello,
                HandshakePhase::ServerHello,
            ]
        );
    }

    #[test]
    fn test_trace_collector() {
        let mut collector = TraceCollector::new(100);
        let id = collector.begin_trace();
        assert_eq!(id, 0);
        collector
            .record_step(
                make_sym_state(0, HandshakePhase::Initial),
                MessageLabel::Tau,
                TraceActionType::InternalComputation,
            )
            .unwrap();
        collector
            .complete_trace(Observable::aborted())
            .unwrap();
        assert_eq!(collector.trace_count(), 1);
        assert_eq!(collector.unique_states_seen(), 1);
        let traces = collector.into_traces();
        assert_eq!(traces.len(), 1);
        assert!(traces[0].is_complete);
    }

    #[test]
    fn test_trace_normalizer_tau_collapse() {
        let mut trace = SymbolicTrace::new(0);
        trace.push_step(make_step(
            0,
            HandshakePhase::Initial,
            MessageLabel::Tau,
            TraceActionType::InternalComputation,
        ));
        trace.push_step(make_step(
            1,
            HandshakePhase::Initial,
            MessageLabel::Tau,
            TraceActionType::InternalComputation,
        ));
        trace.push_step(make_step(
            2,
            HandshakePhase::ClientHello,
            MessageLabel::ClientHello {
                offered_ciphers: BTreeSet::new(),
                version: ProtocolVersion::Tls12,
                extensions: vec![],
            },
            TraceActionType::ProtocolMessage,
        ));
        let normalized = TraceNormalizer::normalize(&trace);
        // Two consecutive Taus should be collapsed into one.
        assert_eq!(normalized.len(), 2);
    }

    #[test]
    fn test_trace_filter() {
        let mut trace = SymbolicTrace::new(0);
        // Internal computation — not inherently relevant.
        trace.push_step(make_step(
            0,
            HandshakePhase::Initial,
            MessageLabel::Tau,
            TraceActionType::InternalComputation,
        ));
        // Protocol message — relevant.
        trace.push_step(make_step(
            1,
            HandshakePhase::ClientHello,
            MessageLabel::ClientHello {
                offered_ciphers: [0x002f].into(),
                version: ProtocolVersion::Tls12,
                extensions: vec![],
            },
            TraceActionType::ProtocolMessage,
        ));
        let filtered = TraceFilter::filter(&trace);
        assert!(filtered.len() >= 1);
        // The protocol message step should always survive.
        assert!(filtered
            .steps
            .iter()
            .any(|s| s.label.label_name() == "CH"));
    }

    #[test]
    fn test_trace_statistics() {
        let traces = vec![make_simple_trace(), make_simple_trace()];
        let stats = TraceStatistics::compute(&traces);
        assert_eq!(stats.total_traces, 2);
        assert_eq!(stats.complete_traces, 2);
        assert_eq!(stats.min_length, 3);
        assert_eq!(stats.max_length, 3);
    }

    #[test]
    fn test_trace_merger() {
        let t1 = make_simple_trace();
        let t2 = make_simple_trace();
        let merged = TraceMerger::merge_traces(vec![t1, t2]);
        // Same fingerprint → merged into one.
        assert_eq!(merged.len(), 1);
    }

    #[test]
    fn test_trace_fingerprint_differs() {
        let t1 = make_simple_trace();
        let mut t2 = SymbolicTrace::new(1);
        t2.push_step(make_step(
            0,
            HandshakePhase::Initial,
            MessageLabel::Tau,
            TraceActionType::InternalComputation,
        ));
        t2.push_step(make_step(
            1,
            HandshakePhase::Alert,
            MessageLabel::Alert {
                level: 2,
                description: 40,
            },
            TraceActionType::ProtocolMessage,
        ));
        assert_ne!(t1.fingerprint(), t2.fingerprint());
    }

    #[test]
    fn test_constraint_aggregator() {
        let mut trace = SymbolicTrace::new(0);
        let mut step = make_step(
            0,
            HandshakePhase::Initial,
            MessageLabel::Tau,
            TraceActionType::BranchDecision,
        );
        step.added_constraint = Some(SymbolicValue::bool_const(true));
        trace.push_step(step);

        let agg = ConstraintAggregator::aggregate(&trace);
        assert!(agg.is_some());
    }

    #[test]
    fn test_canonicalize_state_ids() {
        let mut traces = vec![make_simple_trace()];
        // Original IDs: 0, 1, 2  (already canonical in this case)
        TraceNormalizer::canonicalize_state_ids(&mut traces);
        assert_eq!(traces[0].steps[0].state.id, 0);
        assert_eq!(traces[0].steps[1].state.id, 1);
        assert_eq!(traces[0].steps[2].state.id, 2);
    }
}
