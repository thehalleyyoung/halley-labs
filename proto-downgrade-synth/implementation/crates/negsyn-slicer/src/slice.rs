//! Core PROTOSLICE algorithm (ALG1).
//!
//! Implements the protocol-aware program slicer that extracts negotiation-relevant
//! code from TLS/SSH library IR. Combines taint analysis, dependency analysis,
//! call graph traversal, and vtable resolution to produce a minimal slice
//! capturing all negotiation behaviour.

use std::collections::{HashMap, HashSet, VecDeque, BTreeSet, BTreeMap};
use std::fmt;
use serde::{Serialize, Deserialize};
use indexmap::IndexMap;

use crate::ir::{Module, Function, BasicBlock, Instruction, Value, Type};
use crate::taint::{TaintAnalysis, TaintSource, TaintTag, TaintState, InterprocTaintAnalysis};
use crate::points_to::{AndersonAnalysis, PointsToSet, AbstractLocation};
use crate::callgraph::{CallGraph, CallGraphBuilder, CallSite};
use crate::cfg::{CFG, DominatorTree, PostDominatorTree, ControlDependence, detect_loops};
use crate::dependency::{DependencyGraph, ProgramDependenceGraph, NegotiationDependencyAnalysis};
use crate::vtable::{VTableResolver, CallbackAnalysis};
use crate::{InstructionId, SlicerError, SlicerResult, SlicerConfig, NegotiationPhase};

// ---------------------------------------------------------------------------
// Slice criterion
// ---------------------------------------------------------------------------

/// Defines what the slicer should extract: the negotiation outcome and related paths.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SliceCriterion {
    /// The primary outcome instruction(s) to slice backward from.
    pub outcome_points: Vec<InstructionId>,
    /// Taint source points to slice forward from.
    pub source_points: Vec<InstructionId>,
    /// Negotiation phases to include.
    pub phases: Vec<NegotiationPhase>,
    /// Specific function names to always include.
    pub include_functions: Vec<String>,
    /// Function name patterns to exclude.
    pub exclude_patterns: Vec<String>,
    /// Whether to include control flow dependencies.
    pub include_control_deps: bool,
    /// Whether to include memory dependencies.
    pub include_memory_deps: bool,
    /// Maximum slice size (instructions).
    pub max_slice_size: Option<usize>,
}

impl SliceCriterion {
    /// Create a criterion for cipher suite negotiation outcome.
    pub fn cipher_negotiation() -> Self {
        Self {
            outcome_points: Vec::new(),
            source_points: Vec::new(),
            phases: vec![
                NegotiationPhase::CipherSuiteSelection,
                NegotiationPhase::ClientHello,
                NegotiationPhase::ServerHello,
            ],
            include_functions: vec![
                "ssl3_choose_cipher".into(),
                "tls_choose_cipher".into(),
                "ssl_cipher_list_to_bytes".into(),
            ],
            exclude_patterns: vec![
                "BIO_*".into(),
                "CRYPTO_*".into(),
                "EVP_*".into(),
                "BN_*".into(),
            ],
            include_control_deps: true,
            include_memory_deps: true,
            max_slice_size: None,
        }
    }

    /// Create a criterion for version negotiation.
    pub fn version_negotiation() -> Self {
        Self {
            outcome_points: Vec::new(),
            source_points: Vec::new(),
            phases: vec![
                NegotiationPhase::VersionNegotiation,
                NegotiationPhase::ClientHello,
                NegotiationPhase::ServerHello,
            ],
            include_functions: vec![
                "ssl_set_version_bound".into(),
                "ssl_version_supported".into(),
                "tls1_set_version".into(),
            ],
            exclude_patterns: vec!["BIO_*".into(), "CRYPTO_*".into()],
            include_control_deps: true,
            include_memory_deps: true,
            max_slice_size: None,
        }
    }

    /// Create a criterion targeting a specific function as the negotiation outcome.
    pub fn negotiation_outcome(function_name: &str) -> Self {
        Self {
            outcome_points: Vec::new(),
            source_points: Vec::new(),
            phases: vec![
                NegotiationPhase::CipherSuiteSelection,
                NegotiationPhase::VersionNegotiation,
                NegotiationPhase::ExtensionProcessing,
            ],
            include_functions: vec![function_name.to_string()],
            exclude_patterns: Vec::new(),
            include_control_deps: true,
            include_memory_deps: true,
            max_slice_size: None,
        }
    }

    /// Whether a function should be excluded by the criterion.
    pub fn is_excluded(&self, func_name: &str) -> bool {
        for pattern in &self.exclude_patterns {
            let prefix = pattern.trim_end_matches('*');
            if func_name.starts_with(prefix) {
                return true;
            }
        }
        false
    }

    /// Whether a function should be explicitly included.
    pub fn is_included(&self, func_name: &str) -> bool {
        self.include_functions.iter().any(|f| func_name.contains(f))
    }
}

// ---------------------------------------------------------------------------
// Program slice
// ---------------------------------------------------------------------------

/// The output of the slicer: a subset of the program relevant to negotiation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgramSlice {
    /// Name of the slice.
    pub name: String,
    /// The criterion used.
    pub criterion: SliceCriterion,
    /// Instructions in the slice, grouped by function and block.
    pub instructions: BTreeMap<String, BTreeMap<String, Vec<usize>>>,
    /// Functions included in the slice.
    pub functions: BTreeSet<String>,
    /// Total instruction count.
    pub instruction_count: usize,
    /// Call edges within the slice.
    pub call_edges: Vec<(String, String)>,
    /// Taint sources that reach the outcome.
    pub reaching_sources: Vec<TaintSource>,
    /// Negotiation phases covered.
    pub covered_phases: Vec<NegotiationPhase>,
    /// Metadata.
    pub metadata: SliceMetadata,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SliceMetadata {
    pub original_module_instructions: usize,
    pub slice_reduction_ratio: f64,
    pub analysis_time_ms: u64,
    pub iterations_to_converge: usize,
    pub warnings: Vec<String>,
}

impl ProgramSlice {
    fn new(name: impl Into<String>, criterion: SliceCriterion) -> Self {
        Self {
            name: name.into(),
            criterion,
            instructions: BTreeMap::new(),
            functions: BTreeSet::new(),
            instruction_count: 0,
            call_edges: Vec::new(),
            reaching_sources: Vec::new(),
            covered_phases: Vec::new(),
            metadata: SliceMetadata::default(),
        }
    }

    /// Add an instruction to the slice.
    pub fn add_instruction(&mut self, id: &InstructionId) {
        self.functions.insert(id.function.clone());
        let func_map = self.instructions
            .entry(id.function.clone())
            .or_insert_with(BTreeMap::new);
        let block_vec = func_map.entry(id.block.clone()).or_insert_with(Vec::new);
        if !block_vec.contains(&id.index) {
            block_vec.push(id.index);
            block_vec.sort();
            self.instruction_count += 1;
        }
    }

    /// Add a batch of instructions.
    pub fn add_instructions(&mut self, ids: impl IntoIterator<Item = InstructionId>) {
        for id in ids {
            self.add_instruction(&id);
        }
    }

    /// Whether the slice contains a specific instruction.
    pub fn contains(&self, id: &InstructionId) -> bool {
        self.instructions.get(&id.function)
            .and_then(|blocks| blocks.get(&id.block))
            .map_or(false, |indices| indices.contains(&id.index))
    }

    /// Whether the slice contains any instruction from a function.
    pub fn contains_function(&self, func_name: &str) -> bool {
        self.functions.contains(func_name)
    }

    /// Get all instruction IDs in the slice.
    pub fn all_instructions(&self) -> Vec<InstructionId> {
        let mut result = Vec::with_capacity(self.instruction_count);
        for (func, blocks) in &self.instructions {
            for (block, indices) in blocks {
                for &idx in indices {
                    result.push(InstructionId::new(func, block, idx));
                }
            }
        }
        result
    }

    /// Compute the reduction ratio.
    pub fn reduction_ratio(&self, original_total: usize) -> f64 {
        if original_total == 0 { return 0.0; }
        1.0 - (self.instruction_count as f64 / original_total as f64)
    }

    /// Compute covered negotiation phases from the slice functions.
    pub fn compute_covered_phases(&mut self) {
        let all_phases = [
            NegotiationPhase::ClientHello,
            NegotiationPhase::ServerHello,
            NegotiationPhase::CipherSuiteSelection,
            NegotiationPhase::VersionNegotiation,
            NegotiationPhase::ExtensionProcessing,
            NegotiationPhase::CertificateHandling,
            NegotiationPhase::KeyExchange,
            NegotiationPhase::Finished,
        ];
        self.covered_phases.clear();
        for phase in &all_phases {
            if self.functions.iter().any(|f| phase.matches_function(f)) {
                self.covered_phases.push(*phase);
            }
        }
    }

    /// Serialize to JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

// ---------------------------------------------------------------------------
// Slicing context
// ---------------------------------------------------------------------------

/// Tracks state during the slicing computation.
struct SlicingContext {
    /// Instructions in the current slice.
    relevant: HashSet<InstructionId>,
    /// Worklist of instructions to process.
    worklist: VecDeque<InstructionId>,
    /// Functions we've entered during interprocedural slicing.
    visited_functions: HashSet<String>,
    /// Current call depth.
    call_depth: usize,
    /// Maximum call depth.
    max_call_depth: usize,
    /// Number of iterations.
    iterations: usize,
    /// Maximum iterations.
    max_iterations: usize,
}

impl SlicingContext {
    fn new(max_call_depth: usize, max_iterations: usize) -> Self {
        Self {
            relevant: HashSet::new(),
            worklist: VecDeque::new(),
            visited_functions: HashSet::new(),
            call_depth: 0,
            max_call_depth,
            iterations: 0,
            max_iterations,
        }
    }

    /// Mark an instruction as relevant and add to worklist if new.
    fn mark_relevant(&mut self, id: InstructionId) -> bool {
        if self.relevant.insert(id.clone()) {
            self.worklist.push_back(id);
            true
        } else {
            false
        }
    }

    /// Check if we've exceeded limits.
    fn check_limits(&self) -> SlicerResult<()> {
        if self.iterations > self.max_iterations {
            return Err(SlicerError::AnalysisError {
                phase: "slicing".into(),
                message: format!("Exceeded max iterations: {}", self.max_iterations),
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Protocol-aware slicer (ALG1: PROTOSLICE)
// ---------------------------------------------------------------------------

/// The main PROTOSLICE slicer combining all analyses.
pub struct ProtocolAwareSlicer<'a> {
    module: &'a Module,
    call_graph: CallGraph,
    config: SlicerConfig,
    /// Cached PDGs per function.
    pdgs: HashMap<String, DependencyGraph>,
    /// Cached CFGs per function.
    cfgs: HashMap<String, CFG>,
    /// Taint analysis results.
    taint_results: HashMap<String, HashMap<String, TaintState>>,
    /// VTable resolver.
    vtable_resolver: VTableResolver,
    /// Callback analysis results.
    callback_analysis: CallbackAnalysis,
}

impl<'a> ProtocolAwareSlicer<'a> {
    /// Create a new slicer.
    pub fn new(module: &'a Module, call_graph: &CallGraph) -> Self {
        // We need to rebuild the call graph internally since we can't store a reference
        // to a non-'a reference. Build a fresh one.
        let builder = CallGraphBuilder::new(module).with_indirect_resolution(false);
        let cg = builder.build();

        Self {
            module,
            call_graph: cg,
            config: SlicerConfig::default(),
            pdgs: HashMap::new(),
            cfgs: HashMap::new(),
            taint_results: HashMap::new(),
            vtable_resolver: VTableResolver::new(),
            callback_analysis: CallbackAnalysis::new(),
        }
    }

    /// Create with custom configuration.
    pub fn with_config(module: &'a Module, config: SlicerConfig) -> Self {
        let builder = CallGraphBuilder::new(module).with_indirect_resolution(true);
        let cg = builder.build();

        Self {
            module,
            call_graph: cg,
            config,
            pdgs: HashMap::new(),
            cfgs: HashMap::new(),
            taint_results: HashMap::new(),
            vtable_resolver: VTableResolver::new(),
            callback_analysis: CallbackAnalysis::new(),
        }
    }

    /// Run all pre-analyses.
    pub fn prepare(&mut self) -> SlicerResult<()> {
        log::info!("Preparing slicer analyses for module '{}'", self.module.name);

        // Build CFGs.
        for (fname, func) in &self.module.functions {
            if !func.is_declaration {
                self.cfgs.insert(fname.clone(), CFG::from_function(func));
            }
        }

        // Build PDGs.
        for (fname, func) in &self.module.functions {
            if !func.is_declaration && self.is_relevant_function(func) {
                let pdg = ProgramDependenceGraph::build(func);
                self.pdgs.insert(fname.clone(), pdg);
            }
        }

        // Run taint analysis on relevant functions.
        for (fname, func) in &self.module.functions {
            if !func.is_declaration && self.is_relevant_function(func) {
                let mut taint = TaintAnalysis::new();
                taint.max_iterations = self.config.max_iterations;
                taint.track_memory = self.config.taint_through_memory;
                taint.discover_sources(func);

                match taint.analyze_forward(func) {
                    Ok(results) => { self.taint_results.insert(fname.clone(), results); }
                    Err(e) => {
                        log::warn!("Taint analysis failed for {}: {}", fname, e);
                    }
                }
            }
        }

        // Analyze vtables and callbacks.
        self.vtable_resolver.analyze_module(self.module);
        self.callback_analysis.analyze(self.module);

        log::info!("Preparation complete: {} PDGs, {} taint results",
            self.pdgs.len(), self.taint_results.len());
        Ok(())
    }

    /// Execute the PROTOSLICE algorithm.
    pub fn slice(&mut self, criterion: &SliceCriterion) -> SlicerResult<ProgramSlice> {
        let start = std::time::Instant::now();

        // Phase 1: Identify seed instructions.
        let seeds = self.identify_seeds(criterion);
        log::debug!("Phase 1: {} seed instructions identified", seeds.len());

        // Phase 2: Backward slicing from outcomes.
        let mut ctx = SlicingContext::new(
            self.config.max_call_depth,
            self.config.max_iterations,
        );
        for seed in &seeds {
            ctx.mark_relevant(seed.clone());
        }

        // Phase 3: Worklist-based backward traversal.
        self.backward_slice(&mut ctx, criterion)?;
        log::debug!("Phase 3: {} instructions after backward slice", ctx.relevant.len());

        // Phase 4: Forward expansion from taint sources.
        let taint_additions = self.forward_taint_expansion(criterion);
        for id in &taint_additions {
            ctx.mark_relevant(id.clone());
        }
        log::debug!("Phase 4: {} instructions after taint expansion", ctx.relevant.len());

        // Phase 5: Protocol-specific additions.
        let protocol_additions = self.protocol_specific_additions(criterion, &ctx.relevant);
        for id in &protocol_additions {
            ctx.mark_relevant(id.clone());
        }
        log::debug!("Phase 5: {} instructions after protocol additions", ctx.relevant.len());

        // Phase 6: Completeness fixup — ensure control flow is complete.
        let fixup = self.completeness_fixup(&ctx.relevant);
        for id in &fixup {
            ctx.mark_relevant(id.clone());
        }
        log::debug!("Phase 6: {} instructions after fixup", ctx.relevant.len());

        // Phase 7: Minimize if configured.
        let final_set = if self.config.minimize_slice {
            self.minimize_slice(&ctx.relevant, criterion)
        } else {
            ctx.relevant.clone()
        };

        // Build the ProgramSlice output.
        let mut slice = ProgramSlice::new(
            format!("protoslice_{}", self.module.name),
            criterion.clone(),
        );
        slice.add_instructions(final_set.into_iter());

        // Add call edges within the slice.
        self.add_slice_call_edges(&mut slice);

        // Add taint source info.
        slice.reaching_sources = self.collect_reaching_sources(criterion);

        // Compute covered phases.
        slice.compute_covered_phases();

        // Fill metadata.
        let total = self.module.total_instructions();
        slice.metadata = SliceMetadata {
            original_module_instructions: total,
            slice_reduction_ratio: slice.reduction_ratio(total),
            analysis_time_ms: start.elapsed().as_millis() as u64,
            iterations_to_converge: ctx.iterations,
            warnings: Vec::new(),
        };

        log::info!("PROTOSLICE complete: {} / {} instructions ({:.1}% reduction)",
            slice.instruction_count, total, slice.metadata.slice_reduction_ratio * 100.0);
        Ok(slice)
    }

    /// Identify seed instructions from the criterion.
    fn identify_seeds(&self, criterion: &SliceCriterion) -> Vec<InstructionId> {
        let mut seeds = Vec::new();

        // Add explicit outcome points.
        seeds.extend(criterion.outcome_points.iter().cloned());

        // Add explicit source points.
        seeds.extend(criterion.source_points.iter().cloned());

        // Add return instructions of explicitly included functions.
        for func_name in &criterion.include_functions {
            if let Some(func) = self.module.function(func_name) {
                for (bname, block) in &func.blocks {
                    for (idx, instr) in block.instructions.iter().enumerate() {
                        if matches!(instr, Instruction::Ret { .. }) {
                            seeds.push(InstructionId::new(func_name, bname, idx));
                        }
                    }
                }
            }
            // Also look for partial matches.
            for (fname, func) in &self.module.functions {
                if fname.contains(func_name) && !func.is_declaration {
                    for (bname, block) in &func.blocks {
                        for (idx, instr) in block.instructions.iter().enumerate() {
                            if matches!(instr, Instruction::Ret { .. }) {
                                seeds.push(InstructionId::new(fname, bname, idx));
                            }
                        }
                    }
                }
            }
        }

        // Add instructions in functions matching negotiation phases.
        for phase in &criterion.phases {
            for (fname, func) in &self.module.functions {
                if func.is_declaration { continue; }
                if phase.matches_function(fname) && !criterion.is_excluded(fname) {
                    // Include key instructions: calls, comparisons, returns, stores.
                    for (bname, block) in &func.blocks {
                        for (idx, instr) in block.instructions.iter().enumerate() {
                            let include = matches!(instr,
                                Instruction::Call { .. } | Instruction::ICmp { .. }
                                | Instruction::Ret { .. } | Instruction::Store { .. }
                                | Instruction::CondBr { .. } | Instruction::Switch { .. }
                                | Instruction::Select { .. }
                            );
                            if include {
                                seeds.push(InstructionId::new(fname, bname, idx));
                            }
                        }
                    }
                }
            }
        }

        seeds.sort_by(|a, b| a.to_string().cmp(&b.to_string()));
        seeds.dedup();
        seeds
    }

    /// Backward slicing phase: follow data and control dependencies.
    fn backward_slice(
        &self,
        ctx: &mut SlicingContext,
        criterion: &SliceCriterion,
    ) -> SlicerResult<()> {
        while let Some(id) = ctx.worklist.pop_front() {
            ctx.iterations += 1;
            ctx.check_limits()?;

            // Get dependencies from PDG.
            if let Some(pdg) = self.pdgs.get(&id.function) {
                // Data dependencies.
                for dep_id in pdg.dependencies_of(&id) {
                    if !criterion.is_excluded(&dep_id.function) {
                        ctx.mark_relevant(dep_id.clone());
                    }
                }
            }

            // Interprocedural: if this is in a callee, slice into callers.
            if ctx.call_depth < ctx.max_call_depth {
                let callers = self.call_graph.callers_of(&id.function);
                for caller_site in callers {
                    if !criterion.is_excluded(&caller_site.caller)
                        && !ctx.visited_functions.contains(&caller_site.caller)
                    {
                        ctx.visited_functions.insert(caller_site.caller.clone());
                        ctx.call_depth += 1;
                        ctx.mark_relevant(caller_site.location.clone());
                        ctx.call_depth -= 1;
                    }
                }
            }

            // If this instruction is a call, include the callee function's relevant parts.
            if let Some(func) = self.module.function(&id.function) {
                if let Some(block) = func.blocks.get(&id.block) {
                    if let Some(instr) = block.instructions.get(id.index) {
                        if let Some(callee_name) = instr.called_function_name() {
                            if !criterion.is_excluded(callee_name)
                                && !ctx.visited_functions.contains(callee_name)
                                && ctx.call_depth < ctx.max_call_depth
                            {
                                ctx.visited_functions.insert(callee_name.to_string());
                                ctx.call_depth += 1;
                                self.include_callee_slice(callee_name, ctx, criterion);
                                ctx.call_depth -= 1;
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Include relevant instructions from a callee function.
    fn include_callee_slice(
        &self,
        callee_name: &str,
        ctx: &mut SlicingContext,
        criterion: &SliceCriterion,
    ) {
        if let Some(func) = self.module.function(callee_name) {
            if func.is_declaration { return; }
            // Include return instructions and their data dependencies.
            for (bname, block) in &func.blocks {
                for (idx, instr) in block.instructions.iter().enumerate() {
                    let include = matches!(instr,
                        Instruction::Ret { .. }
                        | Instruction::Call { .. }
                        | Instruction::Store { .. }
                        | Instruction::CondBr { .. }
                    );
                    if include {
                        ctx.mark_relevant(InstructionId::new(callee_name, bname, idx));
                    }
                }
            }
        }
    }

    /// Forward taint expansion: add instructions reachable from taint sources.
    fn forward_taint_expansion(&self, criterion: &SliceCriterion) -> Vec<InstructionId> {
        let mut additions = Vec::new();

        for (fname, taint_states) in &self.taint_results {
            if criterion.is_excluded(fname) { continue; }
            if let Some(func) = self.module.function(fname) {
                for (bname, block) in &func.blocks {
                    if let Some(state) = taint_states.get(bname) {
                        for (idx, instr) in block.instructions.iter().enumerate() {
                            // Include instruction if any operand is tainted.
                            let is_tainted = instr.used_registers().iter().any(|reg| {
                                state.get_register(reg).is_tainted()
                            });
                            if is_tainted {
                                additions.push(InstructionId::new(fname, bname, idx));
                            }
                        }
                    }
                }
            }
        }

        additions
    }

    /// Protocol-specific additions: vtable dispatches, callback chains, ifdef patterns.
    fn protocol_specific_additions(
        &self,
        criterion: &SliceCriterion,
        current: &HashSet<InstructionId>,
    ) -> Vec<InstructionId> {
        let mut additions = Vec::new();

        // Include SSL_METHOD vtable access patterns.
        for (fname, func) in &self.module.functions {
            if func.is_declaration || criterion.is_excluded(fname) { continue; }
            for (bname, block) in &func.blocks {
                for (idx, instr) in block.instructions.iter().enumerate() {
                    let id = InstructionId::new(fname, bname, idx);
                    if current.contains(&id) { continue; }

                    // Include GEP instructions accessing vtable fields.
                    if let Instruction::GetElementPtr { base_ty, .. } = instr {
                        if let Type::NamedStruct(sname) = base_ty {
                            if self.vtable_resolver.layout(sname).is_some() {
                                additions.push(id.clone());
                            }
                        }
                    }

                    // Include calls to callback registration functions.
                    if let Some(callee) = instr.called_function_name() {
                        let callee_lower = callee.to_lowercase();
                        if callee_lower.contains("set_") && (
                            callee_lower.contains("callback")
                            || callee_lower.contains("_cb")
                            || callee_lower.contains("verify")
                            || callee_lower.contains("alpn")
                            || callee_lower.contains("servername")
                        ) {
                            additions.push(id.clone());
                        }
                    }

                    // Include version/cipher checks (ifdef-forest patterns).
                    if let Instruction::ICmp { lhs, rhs, .. } = instr {
                        let involves_version = self.value_involves_version(lhs)
                            || self.value_involves_version(rhs);
                        let involves_cipher = self.value_involves_cipher(lhs)
                            || self.value_involves_cipher(rhs);
                        if involves_version || involves_cipher {
                            additions.push(id.clone());
                        }
                    }
                }
            }
        }

        // Include devirtualization candidates that target negotiation functions.
        let devirted = self.vtable_resolver.devirtualize_candidates(self.module);
        for candidate in &devirted {
            let has_neg_target = candidate.resolved_targets.iter().any(|t| {
                let f = Function::new(t, Type::Void);
                f.is_negotiation_relevant()
            });
            if has_neg_target {
                additions.push(candidate.call_site.clone());
            }
        }

        additions
    }

    /// Ensure completeness: include terminators for blocks with included instructions.
    fn completeness_fixup(&self, current: &HashSet<InstructionId>) -> Vec<InstructionId> {
        let mut additions = Vec::new();

        // Group current instructions by (function, block).
        let mut blocks_present: HashMap<(&str, &str), Vec<usize>> = HashMap::new();
        for id in current {
            blocks_present.entry((&id.function, &id.block))
                .or_default()
                .push(id.index);
        }

        for ((fname, bname), _indices) in &blocks_present {
            if let Some(func) = self.module.function(fname) {
                if let Some(block) = func.blocks.get(*bname) {
                    // Always include the terminator of included blocks.
                    let term_idx = block.instructions.len().saturating_sub(1);
                    if block.instructions.get(term_idx).map_or(false, |i| i.is_terminator()) {
                        let id = InstructionId::new(*fname, *bname, term_idx);
                        if !current.contains(&id) {
                            additions.push(id);
                        }
                    }

                    // Include phi nodes at the start of successor blocks.
                    for succ in &block.successors {
                        if let Some(succ_block) = func.blocks.get(succ) {
                            for (idx, instr) in succ_block.instructions.iter().enumerate() {
                                if matches!(instr, Instruction::Phi { .. }) {
                                    let id = InstructionId::new(*fname, succ, idx);
                                    if !current.contains(&id) {
                                        additions.push(id);
                                    }
                                } else {
                                    break; // Phi nodes are always at the beginning.
                                }
                            }
                        }
                    }
                }
            }
        }

        additions
    }

    /// Minimize the slice by removing non-essential instructions.
    fn minimize_slice(
        &self,
        current: &HashSet<InstructionId>,
        criterion: &SliceCriterion,
    ) -> HashSet<InstructionId> {
        let mut minimized = current.clone();

        // Remove instructions that don't contribute to any outcome.
        let mut to_remove = Vec::new();
        for id in &minimized {
            if let Some(func) = self.module.function(&id.function) {
                if let Some(block) = func.blocks.get(&id.block) {
                    if let Some(instr) = block.instructions.get(id.index) {
                        // Keep terminators, calls, stores, and instructions with tainted operands.
                        let essential = instr.is_terminator()
                            || instr.is_call()
                            || matches!(instr, Instruction::Store { .. })
                            || matches!(instr, Instruction::Ret { .. })
                            || criterion.is_included(&id.function);

                        // Keep if any dependent instruction is in the slice.
                        let has_dependent = if let Some(pdg) = self.pdgs.get(&id.function) {
                            pdg.dependents_of(id).iter().any(|dep| minimized.contains(dep))
                        } else {
                            true // Conservative: keep if we don't have PDG.
                        };

                        if !essential && !has_dependent {
                            to_remove.push(id.clone());
                        }
                    }
                }
            }
        }

        // Only remove up to 20% to avoid over-minimizing.
        let max_removals = minimized.len() / 5;
        for (i, id) in to_remove.iter().enumerate() {
            if i >= max_removals { break; }
            minimized.remove(id);
        }

        minimized
    }

    /// Add call edges within the slice.
    fn add_slice_call_edges(&self, slice: &mut ProgramSlice) {
        for func_name in &slice.functions {
            for callee_site in self.call_graph.callees_of(func_name) {
                if slice.functions.contains(&callee_site.callee) {
                    let edge = (func_name.clone(), callee_site.callee.clone());
                    if !slice.call_edges.contains(&edge) {
                        slice.call_edges.push(edge);
                    }
                }
            }
        }
    }

    /// Collect taint sources that reach the criterion.
    fn collect_reaching_sources(&self, criterion: &SliceCriterion) -> Vec<TaintSource> {
        let mut sources = Vec::new();

        for func_name in &criterion.include_functions {
            if let Some(taint_states) = self.taint_results.get(func_name) {
                for (_block, state) in taint_states {
                    for (_reg, taint_val) in &state.registers {
                        for tag in taint_val.get_tags() {
                            if !sources.contains(&tag.source) {
                                sources.push(tag.source.clone());
                            }
                        }
                    }
                }
            }
        }

        sources
    }

    /// Check whether a value involves version numbers.
    fn value_involves_version(&self, val: &Value) -> bool {
        match val {
            Value::Register(name, _) | Value::GlobalRef(name, _) => {
                let lower = name.to_lowercase();
                lower.contains("version") || lower.contains("tls1") || lower.contains("ssl3")
                    || lower.contains("min_proto") || lower.contains("max_proto")
            }
            Value::IntConst(v, _) => {
                // Known TLS version constants.
                let known = [0x0300, 0x0301, 0x0302, 0x0303, 0x0304, 769, 770, 771, 772];
                known.contains(&(*v as i32))
            }
            _ => false,
        }
    }

    /// Check whether a value involves cipher identifiers.
    fn value_involves_cipher(&self, val: &Value) -> bool {
        match val {
            Value::Register(name, _) | Value::GlobalRef(name, _) => {
                let lower = name.to_lowercase();
                lower.contains("cipher") || lower.contains("suite")
            }
            _ => false,
        }
    }

    /// Whether a function is relevant to the analysis.
    fn is_relevant_function(&self, func: &Function) -> bool {
        if func.is_negotiation_relevant() { return true; }
        for phase in &self.config.target_phases {
            if phase.matches_function(&func.name) { return true; }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::Module;
    use crate::callgraph::CallGraphBuilder;

    #[test]
    fn test_slice_criterion_cipher() {
        let c = SliceCriterion::cipher_negotiation();
        assert!(c.include_functions.contains(&"ssl3_choose_cipher".to_string()));
        assert!(c.phases.contains(&NegotiationPhase::CipherSuiteSelection));
    }

    #[test]
    fn test_slice_criterion_version() {
        let c = SliceCriterion::version_negotiation();
        assert!(c.phases.contains(&NegotiationPhase::VersionNegotiation));
    }

    #[test]
    fn test_slice_criterion_exclusion() {
        let c = SliceCriterion::cipher_negotiation();
        assert!(c.is_excluded("BIO_read"));
        assert!(c.is_excluded("CRYPTO_malloc"));
        assert!(!c.is_excluded("ssl3_choose_cipher"));
    }

    #[test]
    fn test_slice_criterion_inclusion() {
        let c = SliceCriterion::cipher_negotiation();
        assert!(c.is_included("ssl3_choose_cipher"));
        assert!(!c.is_included("BIO_read"));
    }

    #[test]
    fn test_program_slice_basic() {
        let c = SliceCriterion::cipher_negotiation();
        let mut slice = ProgramSlice::new("test", c);
        slice.add_instruction(&InstructionId::new("f", "bb1", 0));
        slice.add_instruction(&InstructionId::new("f", "bb1", 1));
        slice.add_instruction(&InstructionId::new("g", "entry", 0));

        assert_eq!(slice.instruction_count, 3);
        assert_eq!(slice.functions.len(), 2);
        assert!(slice.contains(&InstructionId::new("f", "bb1", 0)));
        assert!(!slice.contains(&InstructionId::new("f", "bb1", 5)));
    }

    #[test]
    fn test_program_slice_dedup() {
        let c = SliceCriterion::cipher_negotiation();
        let mut slice = ProgramSlice::new("test", c);
        slice.add_instruction(&InstructionId::new("f", "bb", 0));
        slice.add_instruction(&InstructionId::new("f", "bb", 0)); // dup
        assert_eq!(slice.instruction_count, 1);
    }

    #[test]
    fn test_program_slice_reduction() {
        let c = SliceCriterion::cipher_negotiation();
        let mut slice = ProgramSlice::new("test", c);
        slice.add_instruction(&InstructionId::new("f", "bb", 0));
        assert!((slice.reduction_ratio(10) - 0.9).abs() < 0.01);
    }

    #[test]
    fn test_slicer_prepare() {
        let module = Module::test_module();
        let cg_builder = CallGraphBuilder::new(&module).with_indirect_resolution(false);
        let cg = cg_builder.build();
        let mut slicer = ProtocolAwareSlicer::new(&module, &cg);
        slicer.prepare().unwrap();
    }

    #[test]
    fn test_slicer_slice() {
        let mut module = Module::test_module();
        module.compute_all_predecessors();
        let cg_builder = CallGraphBuilder::new(&module).with_indirect_resolution(false);
        let cg = cg_builder.build();
        let mut slicer = ProtocolAwareSlicer::new(&module, &cg);
        slicer.prepare().unwrap();

        let criterion = SliceCriterion::negotiation_outcome("ssl3_choose_cipher");
        let result = slicer.slice(&criterion).unwrap();
        assert!(result.instruction_count > 0);
        assert!(result.functions.contains("ssl3_choose_cipher"));
    }

    #[test]
    fn test_slicer_with_config() {
        let module = Module::test_module();
        let config = SlicerConfig {
            max_iterations: 100,
            context_sensitive: false,
            max_call_depth: 3,
            minimize_slice: false,
            ..Default::default()
        };
        let mut slicer = ProtocolAwareSlicer::with_config(&module, config);
        slicer.prepare().unwrap();
    }

    #[test]
    fn test_identify_seeds() {
        let module = Module::test_module();
        let cg_builder = CallGraphBuilder::new(&module).with_indirect_resolution(false);
        let cg = cg_builder.build();
        let slicer = ProtocolAwareSlicer::new(&module, &cg);
        let criterion = SliceCriterion::cipher_negotiation();
        let seeds = slicer.identify_seeds(&criterion);
        assert!(!seeds.is_empty());
    }

    #[test]
    fn test_slice_json() {
        let c = SliceCriterion::cipher_negotiation();
        let mut slice = ProgramSlice::new("test", c);
        slice.add_instruction(&InstructionId::new("f", "bb", 0));
        let json = slice.to_json().unwrap();
        assert!(json.contains("test"));
        assert!(json.contains("\"instruction_count\":1"));
    }

    #[test]
    fn test_all_instructions() {
        let c = SliceCriterion::cipher_negotiation();
        let mut slice = ProgramSlice::new("test", c);
        slice.add_instruction(&InstructionId::new("f", "bb1", 0));
        slice.add_instruction(&InstructionId::new("f", "bb1", 2));
        slice.add_instruction(&InstructionId::new("g", "entry", 1));
        let all = slice.all_instructions();
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn test_covered_phases() {
        let c = SliceCriterion::cipher_negotiation();
        let mut slice = ProgramSlice::new("test", c);
        slice.functions.insert("ssl3_choose_cipher".into());
        slice.functions.insert("tls_construct_client_hello".into());
        slice.compute_covered_phases();
        assert!(slice.covered_phases.contains(&NegotiationPhase::CipherSuiteSelection));
        assert!(slice.covered_phases.contains(&NegotiationPhase::ClientHello));
    }

    #[test]
    fn test_value_involves_version() {
        let module = Module::test_module();
        let cg = CallGraphBuilder::new(&module).with_indirect_resolution(false).build();
        let slicer = ProtocolAwareSlicer::new(&module, &cg);

        assert!(slicer.value_involves_version(&Value::reg("tls1_version", Type::i32())));
        assert!(slicer.value_involves_version(&Value::int(0x0303, 32))); // TLS 1.2
        assert!(!slicer.value_involves_version(&Value::int(42, 32)));
    }

    #[test]
    fn test_slicing_context_limits() {
        let ctx = SlicingContext::new(5, 10);
        assert!(ctx.check_limits().is_ok());

        let mut ctx2 = SlicingContext::new(5, 0);
        ctx2.iterations = 1;
        assert!(ctx2.check_limits().is_err());
    }
}
