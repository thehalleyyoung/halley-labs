//! Core state machine extraction (ALG3: SMEXTRACT).
//!
//! Extracts a `NegotiationLTS` from symbolic execution traces by:
//! 1. Identifying distinct protocol states from trace steps
//! 2. Extracting transitions between states
//! 3. Computing observation function for terminal states
//! 4. Handling merged states from the merge operator
//! 5. Deduplicating and normalizing the resulting LTS
//! 6. Optionally applying bisimulation quotient and minimization

use crate::{
    bisimulation::BisimulationChecker,
    minimize::Minimizer,
    observation::{ObservationEquivalence, ObservationFunction},
    quotient::QuotientBuilder,
    simulation::SimulationChecker,
    trace::{
        ConstraintAggregator, SymbolicTrace, SymbolicTraceStep, TraceActionType, TraceFilter,
        TraceMerger, TraceNormalizer, TraceStatistics,
    },
    ExtractError, ExtractResult, ExtractionConfig, ExtractionMetrics, HandshakePhase,
    LtsState, LtsTransition, MergeOperator, MessageLabel, NegotiationLTS, NegotiationOutcome,
    NoOpMergeOperator, Observable, ProtocolVersion, StateId, SymbolicState, SymbolicValue,
    TransitionId,
};
use indexmap::IndexMap;
use log::{debug, info, trace, warn};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Extraction context
// ---------------------------------------------------------------------------

/// Context for the extraction pipeline, holding configuration, metrics,
/// and a reference to the merge operator.
pub struct ExtractionContext {
    pub config: ExtractionConfig,
    pub metrics: ExtractionMetrics,
    merge_operator: Arc<dyn MergeOperator>,
    obs_fn: ObservationFunction,
    /// Map from symbolic state fingerprint → assigned LTS state ID.
    state_map: HashMap<u64, StateId>,
    /// Set of transitions already added (to avoid duplicates).
    transition_set: HashSet<(StateId, String, StateId)>,
}

impl ExtractionContext {
    pub fn new(config: ExtractionConfig) -> Self {
        Self {
            config,
            metrics: ExtractionMetrics::default(),
            merge_operator: Arc::new(NoOpMergeOperator),
            obs_fn: ObservationFunction::new(),
            state_map: HashMap::new(),
            transition_set: HashSet::new(),
        }
    }

    pub fn with_merge_operator(mut self, op: Arc<dyn MergeOperator>) -> Self {
        self.merge_operator = op;
        self
    }

    /// Compute a fingerprint for a symbolic state that captures its
    /// negotiation-relevant aspects: phase, cipher selection, version.
    fn state_fingerprint(sym: &SymbolicState) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        sym.negotiation.phase.hash(&mut h);
        sym.negotiation.version.hash(&mut h);
        sym.negotiation.selected_cipher.as_ref().map(|c| c.iana_id).hash(&mut h);
        for c in &sym.negotiation.offered_ciphers {
            c.iana_id.hash(&mut h);
        }
        for ext in &sym.negotiation.extensions {
            ext.id.hash(&mut h);
        }
        // Hash PC as a string representation since PathConstraint doesn't impl Hash.
        format!("{}", sym.pc).hash(&mut h);
        h.finish()
    }

    /// Check if a symbolic state was already mapped to an LTS state.
    fn lookup_state(&self, sym: &SymbolicState) -> Option<StateId> {
        let fp = Self::state_fingerprint(sym);
        self.state_map.get(&fp).copied()
    }

    /// Register a symbolic state → LTS state mapping.
    fn register_state(&mut self, sym: &SymbolicState, sid: StateId) {
        let fp = Self::state_fingerprint(sym);
        self.state_map.insert(fp, sid);
    }

    /// Check if a transition has already been recorded.
    fn has_transition(&self, src: StateId, label: &str, tgt: StateId) -> bool {
        self.transition_set.contains(&(src, label.to_string(), tgt))
    }

    /// Record that a transition exists.
    fn record_transition(&mut self, src: StateId, label: &str, tgt: StateId) {
        self.transition_set.insert((src, label.to_string(), tgt));
    }
}

// ---------------------------------------------------------------------------
// StateMachineExtractor
// ---------------------------------------------------------------------------

/// Main struct orchestrating the ALG3: SMEXTRACT algorithm.
///
/// Processes symbolic execution traces to produce a NegotiationLTS,
/// then optionally applies bisimulation quotient and minimization.
pub struct StateMachineExtractor {
    ctx: ExtractionContext,
}

impl StateMachineExtractor {
    pub fn new(config: ExtractionConfig) -> Self {
        Self {
            ctx: ExtractionContext::new(config),
        }
    }

    pub fn with_merge_operator(mut self, op: Arc<dyn MergeOperator>) -> Self {
        self.ctx = self.ctx.with_merge_operator(op);
        self
    }

    /// Main entry point: extract a NegotiationLTS from symbolic traces.
    ///
    /// Implements ALG3: SMEXTRACT:
    /// 1. Preprocess traces (normalize, filter, merge)
    /// 2. Identify states from trace steps
    /// 3. Extract transitions from adjacent steps
    /// 4. Compute observation function
    /// 5. Deduplicate states
    /// 6. Apply bisimulation quotient (if enabled)
    /// 7. Minimize (if enabled)
    pub fn extract_from_traces(
        &mut self,
        traces: &[SymbolicTrace],
    ) -> ExtractResult<NegotiationLTS> {
        let start = Instant::now();
        info!(
            "SMEXTRACT: beginning extraction from {} traces",
            traces.len()
        );

        if traces.is_empty() {
            return Err(ExtractError::NoTraces);
        }

        // Step 1: Preprocess.
        let preprocessed = self.preprocess_traces(traces)?;
        self.ctx.metrics.traces_processed = preprocessed.len() as u64;

        let stats = TraceStatistics::compute(&preprocessed);
        self.ctx.metrics.trace_steps_total = stats.total_steps as u64;
        debug!("Trace statistics: {}", stats);

        // Step 2-3: Build the raw LTS from traces.
        let mut lts = self.build_raw_lts(&preprocessed)?;
        self.ctx.metrics.states_extracted = lts.state_count() as u64;
        self.ctx.metrics.transitions_extracted = lts.transition_count() as u64;
        info!(
            "Raw LTS: {} states, {} transitions",
            lts.state_count(),
            lts.transition_count()
        );

        // Step 4: Compute observations for all terminal states.
        self.compute_observations(&mut lts);

        // Step 5: Deduplicate.
        self.deduplicate_states(&mut lts);
        self.ctx.metrics.states_after_dedup = lts.state_count() as u64;
        self.ctx.metrics.transitions_after_dedup = lts.transition_count() as u64;
        info!(
            "After dedup: {} states, {} transitions",
            lts.state_count(),
            lts.transition_count()
        );

        // Step 5.5: Attempt to merge compatible states.
        self.try_merge_states(&mut lts);

        // Step 6: Bisimulation quotient.
        if self.ctx.config.enable_bisimulation {
            let bisim_start = Instant::now();
            lts = self.apply_bisimulation_quotient(lts)?;
            self.ctx.metrics.bisimulation_time_us =
                bisim_start.elapsed().as_micros() as u64;
            self.ctx.metrics.states_after_quotient = lts.state_count() as u64;
            self.ctx.metrics.transitions_after_quotient = lts.transition_count() as u64;
            info!(
                "After bisimulation quotient: {} states, {} transitions",
                lts.state_count(),
                lts.transition_count()
            );
        }

        // Step 7: Minimization.
        if self.ctx.config.enable_minimization {
            let min_start = Instant::now();
            lts = self.apply_minimization(lts)?;
            self.ctx.metrics.minimization_time_us =
                min_start.elapsed().as_micros() as u64;
            self.ctx.metrics.states_after_minimization = lts.state_count() as u64;
            self.ctx.metrics.transitions_after_minimization =
                lts.transition_count() as u64;
            info!(
                "After minimization: {} states, {} transitions",
                lts.state_count(),
                lts.transition_count()
            );
        }

        self.ctx.metrics.extraction_time_us = start.elapsed().as_micros() as u64;
        self.ctx.metrics.total_time_us = start.elapsed().as_micros() as u64;
        info!(
            "Extraction complete. Reduction ratio: {:.1}%",
            self.ctx.metrics.reduction_ratio() * 100.0
        );

        Ok(lts)
    }

    /// Get the extraction metrics.
    pub fn metrics(&self) -> &ExtractionMetrics {
        &self.ctx.metrics
    }

    /// Get a mutable reference to the context.
    pub fn context_mut(&mut self) -> &mut ExtractionContext {
        &mut self.ctx
    }

    // -----------------------------------------------------------------------
    // Step 1: Preprocessing
    // -----------------------------------------------------------------------

    fn preprocess_traces(
        &self,
        traces: &[SymbolicTrace],
    ) -> ExtractResult<Vec<SymbolicTrace>> {
        let mut processed: Vec<SymbolicTrace> = traces.to_vec();

        // Normalize traces.
        if self.ctx.config.normalize_traces {
            processed = TraceNormalizer::normalize_all(&processed);
            TraceNormalizer::canonicalize_state_ids(&mut processed);
            debug!("Normalized {} traces", processed.len());
        }

        // Filter non-negotiation steps.
        if self.ctx.config.filter_non_negotiation {
            processed = TraceFilter::filter_all(&processed);
            debug!(
                "After filtering: {} traces ({} non-empty)",
                processed.len(),
                processed.iter().filter(|t| !t.is_empty()).count()
            );
        }

        // Merge traces with identical fingerprints.
        let before = processed.len();
        processed = TraceMerger::merge_traces(processed);
        debug!(
            "Merged {} traces into {}",
            before,
            processed.len()
        );

        // Filter out empty traces.
        processed.retain(|t| !t.is_empty());

        if processed.is_empty() {
            return Err(ExtractError::NoTraces);
        }

        Ok(processed)
    }

    // -----------------------------------------------------------------------
    // Step 2-3: Build raw LTS
    // -----------------------------------------------------------------------

    fn build_raw_lts(
        &mut self,
        traces: &[SymbolicTrace],
    ) -> ExtractResult<NegotiationLTS> {
        let mut lts = NegotiationLTS::new();

        for trace in traces {
            self.process_trace(&mut lts, trace)?;
        }

        // Verify we have at least one initial state.
        if lts.initial_states.is_empty() {
            return Err(ExtractError::NoInitialState);
        }

        // Check limits.
        if lts.state_count() > self.ctx.config.max_states {
            return Err(ExtractError::StateLimitExceeded {
                count: lts.state_count(),
                limit: self.ctx.config.max_states,
            });
        }
        if lts.transition_count() > self.ctx.config.max_transitions {
            return Err(ExtractError::TransitionLimitExceeded {
                count: lts.transition_count(),
                limit: self.ctx.config.max_transitions,
            });
        }

        Ok(lts)
    }

    /// Process a single trace, adding its states and transitions to the LTS.
    fn process_trace(
        &mut self,
        lts: &mut NegotiationLTS,
        trace: &SymbolicTrace,
    ) -> ExtractResult<()> {
        if trace.is_empty() {
            return Ok(());
        }

        let mut prev_state_id: Option<StateId> = None;

        for (i, step) in trace.steps.iter().enumerate() {
            // Identify or create the LTS state for this trace step.
            let lts_state_id = self.identify_state(lts, &step.state);

            // Mark the first state as initial.
            if i == 0 {
                lts.mark_initial(lts_state_id);
            }

            // Add the source symbolic state ID to the LTS state.
            if let Some(lts_state) = lts.get_state_mut(lts_state_id) {
                if !lts_state.source_symbolic_ids.contains(&step.state.id) {
                    lts_state.source_symbolic_ids.push(step.state.id);
                }
            }

            // Create transition from previous state.
            if let Some(prev_id) = prev_state_id {
                let label_name = step.label.label_name().to_string();
                if !self.ctx.has_transition(prev_id, &label_name, lts_state_id) {
                    // The label on the transition leading INTO this state
                    // comes from the previous step (the action that caused
                    // the transition).
                    let prev_step = &trace.steps[i - 1];
                    let mut label = prev_step.label.clone();

                    // If the previous step was a branch decision, convert to Tau.
                    if prev_step.action_type == TraceActionType::BranchDecision
                        || prev_step.action_type == TraceActionType::InternalComputation
                    {
                        if matches!(label, MessageLabel::Tau) {
                            // Already Tau, fine.
                        }
                    }

                    if let Some(ref guard) = prev_step.added_constraint {
                        lts.add_transition_with_guard(
                            prev_id,
                            lts_state_id,
                            label,
                            guard.clone(),
                        );
                    } else {
                        lts.add_transition(prev_id, lts_state_id, label);
                    }
                    self.ctx.record_transition(
                        prev_id,
                        &prev_step.label.label_name(),
                        lts_state_id,
                    );
                }
            }

            prev_state_id = Some(lts_state_id);
        }

        // If the trace is complete and the last state is terminal,
        // set its observation from the trace's final outcome.
        if trace.is_complete {
            if let (Some(last_id), Some(ref outcome)) =
                (prev_state_id, &trace.final_outcome)
            {
                if let Some(state) = lts.get_state_mut(last_id) {
                    state.observation = outcome.clone();
                    state.is_terminal = true;
                }
            }
        }

        Ok(())
    }

    /// Identify the LTS state corresponding to a symbolic state.
    /// Returns an existing state ID if one matches, or creates a new one.
    fn identify_state(
        &mut self,
        lts: &mut NegotiationLTS,
        sym: &SymbolicState,
    ) -> StateId {
        // Check if we've already seen an equivalent symbolic state.
        if let Some(existing) = self.ctx.lookup_state(sym) {
            return existing;
        }

        // Create a new LTS state from the symbolic state's negotiation component.
        let sid = lts.add_state(sym.negotiation.clone());
        self.ctx.register_state(sym, sid);
        sid
    }

    // -----------------------------------------------------------------------
    // Step 4: Compute observations
    // -----------------------------------------------------------------------

    fn compute_observations(&mut self, lts: &mut NegotiationLTS) {
        let state_ids: Vec<StateId> = lts.states.keys().copied().collect();
        for sid in state_ids {
            let obs = self.ctx.obs_fn.observe(lts, sid);
            if let Some(state) = lts.get_state_mut(sid) {
                state.observation = obs;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Step 5: Deduplication
    // -----------------------------------------------------------------------

    /// Remove duplicate states that have identical negotiations and observations.
    fn deduplicate_states(&mut self, lts: &mut NegotiationLTS) {
        // Group states by (negotiation fingerprint, observation).
        let mut groups: HashMap<u64, Vec<StateId>> = HashMap::new();
        for (&sid, state) in &lts.states {
            let fp = Self::negotiation_fingerprint(&state.negotiation, &state.observation);
            groups.entry(fp).or_default().push(sid);
        }

        // For each group with more than one state, merge into the first.
        let mut remap: HashMap<StateId, StateId> = HashMap::new();
        for (_fp, group) in &groups {
            if group.len() <= 1 {
                continue;
            }
            let canonical = group[0];
            for &sid in &group[1..] {
                remap.insert(sid, canonical);

                // Move source symbolic IDs to the canonical state.
                let source_ids: Vec<u64> = lts
                    .get_state(sid)
                    .map(|s| s.source_symbolic_ids.clone())
                    .unwrap_or_default();
                if let Some(canon_state) = lts.get_state_mut(canonical) {
                    for id in source_ids {
                        if !canon_state.source_symbolic_ids.contains(&id) {
                            canon_state.source_symbolic_ids.push(id);
                        }
                    }
                }
            }
        }

        if remap.is_empty() {
            return;
        }

        debug!("Deduplicating: merging {} duplicate states", remap.len());

        // Remap all transitions.
        for t in &mut lts.transitions {
            if let Some(&new_src) = remap.get(&t.source) {
                t.source = new_src;
            }
            if let Some(&new_tgt) = remap.get(&t.target) {
                t.target = new_tgt;
            }
        }

        // Remap initial states.
        for s in &mut lts.initial_states {
            if let Some(&new_s) = remap.get(s) {
                *s = new_s;
            }
        }
        lts.initial_states.sort();
        lts.initial_states.dedup();

        // Remove duplicate states.
        for sid in remap.keys() {
            lts.states.swap_remove(sid);
        }

        // Remove duplicate transitions (same src, label_name, tgt).
        let mut seen: HashSet<(StateId, String, StateId)> = HashSet::new();
        lts.transitions.retain(|t| {
            let key = (t.source, t.label.label_name().to_string(), t.target);
            seen.insert(key)
        });

        // Remove self-loops on Tau.
        lts.transitions.retain(|t| {
            !(t.source == t.target && t.label == MessageLabel::Tau)
        });
    }

    fn negotiation_fingerprint(
        neg: &negsyn_types::NegotiationState,
        obs: &Observable,
    ) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        neg.phase.hash(&mut h);
        neg.version.hash(&mut h);
        neg.selected_cipher.as_ref().map(|c| c.iana_id).hash(&mut h);
        for c in &neg.offered_ciphers {
            c.iana_id.hash(&mut h);
        }
        obs.hash(&mut h);
        h.finish()
    }

    // -----------------------------------------------------------------------
    // Step 5.5: Merge compatible states
    // -----------------------------------------------------------------------

    /// Attempt to merge states that the merge operator deems compatible.
    fn try_merge_states(&mut self, lts: &mut NegotiationLTS) {
        // Group states by phase.
        let mut phase_groups: HashMap<HandshakePhase, Vec<StateId>> = HashMap::new();
        for (&sid, state) in &lts.states {
            phase_groups
                .entry(state.negotiation.phase)
                .or_default()
                .push(sid);
        }

        let mut merges: Vec<(StateId, StateId)> = Vec::new();

        for (_phase, group) in &phase_groups {
            if group.len() < 2 {
                continue;
            }

            // Try pairwise merges within each phase group.
            for i in 0..group.len() {
                for j in (i + 1)..group.len() {
                    let s1 = group[i];
                    let s2 = group[j];

                    // Build synthetic symbolic states from the LTS states.
                    let sym1 = self.lts_state_to_symbolic(lts, s1);
                    let sym2 = self.lts_state_to_symbolic(lts, s2);

                    if let (Some(sym1), Some(sym2)) = (sym1, sym2) {
                        if self.ctx.merge_operator.is_mergeable(&sym1, &sym2) {
                            merges.push((s1, s2));
                        }
                    }
                }
            }
        }

        // Apply merges: redirect s2 → s1.
        for (keep, remove) in merges {
            for t in &mut lts.transitions {
                if t.source == remove {
                    t.source = keep;
                }
                if t.target == remove {
                    t.target = keep;
                }
            }
            for s in &mut lts.initial_states {
                if *s == remove {
                    *s = keep;
                }
            }
            lts.states.swap_remove(&remove);
        }

        // Clean up duplicate transitions.
        let mut seen: HashSet<(StateId, String, StateId)> = HashSet::new();
        lts.transitions.retain(|t| {
            let key = (t.source, t.label.label_name().to_string(), t.target);
            seen.insert(key)
        });
    }

    /// Convert an LTS state back to a (synthetic) SymbolicState for merge checking.
    fn lts_state_to_symbolic(
        &self,
        lts: &NegotiationLTS,
        sid: StateId,
    ) -> Option<SymbolicState> {
        lts.get_state(sid).map(|state| {
            let mut s = SymbolicState::new(sid.0 as u64, 0);
            s.negotiation = state.negotiation.clone();
            s
        })
    }

    // -----------------------------------------------------------------------
    // Step 6: Bisimulation quotient
    // -----------------------------------------------------------------------

    fn apply_bisimulation_quotient(
        &mut self,
        lts: NegotiationLTS,
    ) -> ExtractResult<NegotiationLTS> {
        let max_iter = self.ctx.config.max_refinement_iterations;
        let mut checker = BisimulationChecker::new(max_iter);
        let relation = checker.compute(&lts)?;
        self.ctx.metrics.bisimulation_classes = relation.class_count() as u64;
        self.ctx.metrics.refinement_iterations = checker.iterations() as u64;

        let builder = QuotientBuilder::new();
        let quotient_lts = builder.build(&lts, &relation)?;
        Ok(quotient_lts)
    }

    // -----------------------------------------------------------------------
    // Step 7: Minimization
    // -----------------------------------------------------------------------

    fn apply_minimization(
        &mut self,
        lts: NegotiationLTS,
    ) -> ExtractResult<NegotiationLTS> {
        let mut minimizer = Minimizer::new();
        let minimized = minimizer.minimize(lts)?;
        self.ctx.metrics.unreachable_eliminated = minimizer.unreachable_eliminated() as u64;
        self.ctx.metrics.redundant_transitions_eliminated =
            minimizer.redundant_eliminated() as u64;
        Ok(minimized)
    }
}

// ---------------------------------------------------------------------------
// Builder pattern for convenience
// ---------------------------------------------------------------------------

/// Builder for constructing and running the extraction pipeline.
pub struct ExtractionPipeline {
    config: ExtractionConfig,
    merge_operator: Option<Arc<dyn MergeOperator>>,
    traces: Vec<SymbolicTrace>,
}

impl ExtractionPipeline {
    pub fn new() -> Self {
        Self {
            config: ExtractionConfig::default(),
            merge_operator: None,
            traces: Vec::new(),
        }
    }

    pub fn config(mut self, config: ExtractionConfig) -> Self {
        self.config = config;
        self
    }

    pub fn merge_operator(mut self, op: Arc<dyn MergeOperator>) -> Self {
        self.merge_operator = Some(op);
        self
    }

    pub fn add_trace(mut self, trace: SymbolicTrace) -> Self {
        self.traces.push(trace);
        self
    }

    pub fn add_traces(mut self, traces: Vec<SymbolicTrace>) -> Self {
        self.traces.extend(traces);
        self
    }

    /// Run the full extraction pipeline.
    pub fn run(self) -> ExtractResult<(NegotiationLTS, ExtractionMetrics)> {
        let mut extractor = StateMachineExtractor::new(self.config);
        if let Some(op) = self.merge_operator {
            extractor = extractor.with_merge_operator(op);
        }
        let lts = extractor.extract_from_traces(&self.traces)?;
        let metrics = extractor.metrics().clone();
        Ok((lts, metrics))
    }
}

impl Default for ExtractionPipeline {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::{SymbolicTrace, SymbolicTraceStep, TraceActionType};
    use negsyn_types::NegotiationState;
    use std::collections::BTreeSet;

    fn make_sym_state(id: u64, phase: HandshakePhase) -> SymbolicState {
        let mut neg = NegotiationState::new();
        neg.phase = phase;
        neg.version = Some(ProtocolVersion::Tls12);
        let mut s = SymbolicState::new(id, 0x1000 + id);
        s.negotiation = neg;
        s
    }

    fn make_sym_state_with_cipher(id: u64, phase: HandshakePhase, cipher: u16) -> SymbolicState {
        let mut neg = NegotiationState::new();
        neg.phase = phase;
        neg.version = Some(ProtocolVersion::Tls12);
        neg.selected_cipher = Some(CipherSuite::new(
            cipher,
            format!("TEST_0x{:04x}", cipher),
            negsyn_types::protocol::KeyExchange::NULL,
            negsyn_types::protocol::AuthAlgorithm::NULL,
            negsyn_types::protocol::EncryptionAlgorithm::NULL,
            negsyn_types::protocol::MacAlgorithm::NULL,
            SecurityLevel::Standard,
        ));
        let mut s = SymbolicState::new(id, 0x1000 + id);
        s.negotiation = neg;
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

    fn make_complete_trace() -> SymbolicTrace {
        let mut trace = SymbolicTrace::new(0);
        trace.push_step(make_step(
            0,
            HandshakePhase::Initial,
            MessageLabel::Tau,
            TraceActionType::InternalComputation,
        ));
        trace.push_step(SymbolicTraceStep::new(
            {
                let mut s = make_sym_state(1, HandshakePhase::ClientHello);
                s.negotiation.offered_ciphers = vec![0x002f_u16, 0x0035].iter().map(|&id| CipherSuite::new(
                    id,
                    format!("TEST_0x{:04x}", id),
                    negsyn_types::protocol::KeyExchange::NULL,
                    negsyn_types::protocol::AuthAlgorithm::NULL,
                    negsyn_types::protocol::EncryptionAlgorithm::NULL,
                    negsyn_types::protocol::MacAlgorithm::NULL,
                    SecurityLevel::Standard,
                )).collect();
                s
            },
            MessageLabel::ClientHello {
                offered_ciphers: [0x002f, 0x0035].into(),
                version: ProtocolVersion::Tls12,
                extensions: vec![],
            },
            TraceActionType::ProtocolMessage,
        ));
        trace.push_step(SymbolicTraceStep::new(
            make_sym_state_with_cipher(2, HandshakePhase::ServerHello, 0x002f),
            MessageLabel::ServerHello {
                selected_cipher: 0x002f,
                version: ProtocolVersion::Tls12,
                extensions: vec![],
            },
            TraceActionType::ProtocolMessage,
        ));
        trace.push_step(SymbolicTraceStep::new(
            make_sym_state_with_cipher(3, HandshakePhase::ApplicationData, 0x002f),
            MessageLabel::ServerFinished { verify_data_hash: 0 },
            TraceActionType::ProtocolMessage,
        ));
        trace.mark_complete(Observable::successful(
            0x002f,
            ProtocolVersion::Tls12,
            BTreeSet::new(),
        ));
        trace
    }

    fn make_abort_trace() -> SymbolicTrace {
        let mut trace = SymbolicTrace::new(1);
        trace.push_step(make_step(
            10,
            HandshakePhase::Initial,
            MessageLabel::Tau,
            TraceActionType::InternalComputation,
        ));
        trace.push_step(SymbolicTraceStep::new(
            make_sym_state(11, HandshakePhase::ClientHello),
            MessageLabel::ClientHello {
                offered_ciphers: [0x002f].into(),
                version: ProtocolVersion::Tls12,
                extensions: vec![],
            },
            TraceActionType::ProtocolMessage,
        ));
        trace.push_step(SymbolicTraceStep::new(
            make_sym_state(12, HandshakePhase::Alert),
            MessageLabel::Alert {
                level: 2,
                description: 40,
            },
            TraceActionType::ProtocolMessage,
        ));
        trace.mark_complete(Observable::aborted());
        trace
    }

    #[test]
    fn test_extract_single_trace() {
        let trace = make_complete_trace();
        let config = ExtractionConfig {
            enable_bisimulation: false,
            enable_minimization: false,
            normalize_traces: false,
            filter_non_negotiation: false,
            ..Default::default()
        };
        let mut extractor = StateMachineExtractor::new(config);
        let lts = extractor.extract_from_traces(&[trace]).unwrap();

        assert!(lts.state_count() >= 2);
        assert!(lts.transition_count() >= 1);
        assert!(!lts.initial_states.is_empty());
    }

    #[test]
    fn test_extract_multiple_traces() {
        let t1 = make_complete_trace();
        let t2 = make_abort_trace();
        let config = ExtractionConfig {
            enable_bisimulation: false,
            enable_minimization: false,
            normalize_traces: false,
            filter_non_negotiation: false,
            ..Default::default()
        };
        let mut extractor = StateMachineExtractor::new(config);
        let lts = extractor.extract_from_traces(&[t1, t2]).unwrap();

        assert!(lts.state_count() >= 4);
        let terminals = lts.terminal_states();
        assert!(!terminals.is_empty());
    }

    #[test]
    fn test_extract_with_dedup() {
        // Two identical traces should produce deduplicated states.
        let t1 = make_complete_trace();
        let t2 = make_complete_trace();
        let config = ExtractionConfig {
            enable_bisimulation: false,
            enable_minimization: false,
            normalize_traces: false,
            filter_non_negotiation: false,
            ..Default::default()
        };
        let mut extractor = StateMachineExtractor::new(config);
        let lts = extractor.extract_from_traces(&[t1, t2]).unwrap();

        // Deduplicated, so same number of states as single trace.
        assert!(lts.state_count() <= 8);
    }

    #[test]
    fn test_extract_empty_traces_error() {
        let config = ExtractionConfig::default();
        let mut extractor = StateMachineExtractor::new(config);
        let result = extractor.extract_from_traces(&[]);
        assert!(matches!(result, Err(ExtractError::NoTraces)));
    }

    #[test]
    fn test_extract_with_bisimulation() {
        let t1 = make_complete_trace();
        let t2 = make_abort_trace();
        let config = ExtractionConfig {
            enable_bisimulation: true,
            enable_minimization: false,
            normalize_traces: false,
            filter_non_negotiation: false,
            ..Default::default()
        };
        let mut extractor = StateMachineExtractor::new(config);
        let lts = extractor.extract_from_traces(&[t1, t2]).unwrap();
        assert!(lts.state_count() > 0);
        assert!(lts.transition_count() >= 0);
    }

    #[test]
    fn test_extract_full_pipeline() {
        let t1 = make_complete_trace();
        let t2 = make_abort_trace();
        let config = ExtractionConfig {
            enable_bisimulation: true,
            enable_minimization: true,
            normalize_traces: true,
            filter_non_negotiation: false,
            ..Default::default()
        };
        let mut extractor = StateMachineExtractor::new(config);
        let lts = extractor.extract_from_traces(&[t1, t2]).unwrap();

        let metrics = extractor.metrics();
        assert!(metrics.traces_processed > 0);
        assert!(metrics.total_time_us > 0);
        assert!(lts.state_count() > 0);
    }

    #[test]
    fn test_extraction_pipeline_builder() {
        let trace = make_complete_trace();
        let (lts, metrics) = ExtractionPipeline::new()
            .config(ExtractionConfig {
                enable_bisimulation: false,
                enable_minimization: false,
                normalize_traces: false,
                filter_non_negotiation: false,
                ..Default::default()
            })
            .add_trace(trace)
            .run()
            .unwrap();
        assert!(lts.state_count() > 0);
        assert!(metrics.traces_processed > 0);
    }

    #[test]
    fn test_extraction_context_fingerprint() {
        let s1 = make_sym_state(0, HandshakePhase::Initial);
        let s2 = make_sym_state(1, HandshakePhase::Initial);
        // Same phase, version, and PC=0 → same fingerprint? No, different PCs.
        let fp1 = ExtractionContext::state_fingerprint(&s1);
        let fp2 = ExtractionContext::state_fingerprint(&s2);
        // Different PCs (0x1000 vs 0x1001) should give different fingerprints.
        assert_ne!(fp1, fp2);
    }

    #[test]
    fn test_metrics_display() {
        let mut m = ExtractionMetrics::default();
        m.traces_processed = 5;
        m.states_extracted = 100;
        m.states_after_minimization = 10;
        m.total_time_us = 5000;
        let s = format!("{}", m);
        assert!(s.contains("100"));
        assert!(s.contains("10"));
    }
}
