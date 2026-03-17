//! Taint analysis for protocol negotiation fields.
//!
//! Implements forward and backward taint propagation to track how negotiation
//! values (cipher suites, protocol versions, extensions) flow through program code.

use std::collections::{HashMap, HashSet, VecDeque, BTreeSet};
use std::fmt;
use serde::{Serialize, Deserialize};
use indexmap::IndexMap;
use smallvec::SmallVec;

use crate::ir::{Module, Function, BasicBlock, Instruction, Value, Type, BinOp};
use crate::{InstructionId, SlicerError, SlicerResult, SlicerConfig, NegotiationPhase};

// ---------------------------------------------------------------------------
// Taint sources
// ---------------------------------------------------------------------------

/// Origin of a taint — the kind of negotiation data that taints a value.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TaintSource {
    /// A struct field holding offered/selected cipher suites.
    NegotiationField { struct_name: String, field_index: u32 },
    /// The cipher suite list (client or server).
    CipherSuiteList { is_client: bool },
    /// Protocol version field (major, minor).
    VersionField { field: String },
    /// Extension data buffer.
    ExtensionData { extension_type: u16 },
    /// Return value of a selection function (e.g., ssl3_choose_cipher).
    SelectionFunction { function_name: String },
    /// Return value of a callback.
    CallbackReturn { callback_name: String },
    /// Global configuration value.
    ConfigValue { name: String },
    /// Parameter of a protocol-entry function.
    EntryParameter { function: String, param_index: usize },
}

impl TaintSource {
    /// Whether this taint source is cipher-suite related.
    pub fn is_cipher_related(&self) -> bool {
        matches!(self, TaintSource::CipherSuiteList { .. }
            | TaintSource::SelectionFunction { .. })
            || match self {
                TaintSource::NegotiationField { struct_name, .. } =>
                    struct_name.contains("cipher") || struct_name.contains("CIPHER"),
                _ => false,
            }
    }

    /// Whether this taint source is version-related.
    pub fn is_version_related(&self) -> bool {
        matches!(self, TaintSource::VersionField { .. })
            || match self {
                TaintSource::NegotiationField { struct_name, .. } =>
                    struct_name.contains("version") || struct_name.contains("VERSION"),
                _ => false,
            }
    }
}

impl fmt::Display for TaintSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TaintSource::NegotiationField { struct_name, field_index } =>
                write!(f, "field({}.{})", struct_name, field_index),
            TaintSource::CipherSuiteList { is_client } =>
                write!(f, "cipher_list({})", if *is_client { "client" } else { "server" }),
            TaintSource::VersionField { field } =>
                write!(f, "version({})", field),
            TaintSource::ExtensionData { extension_type } =>
                write!(f, "ext(0x{:04x})", extension_type),
            TaintSource::SelectionFunction { function_name } =>
                write!(f, "select_fn({})", function_name),
            TaintSource::CallbackReturn { callback_name } =>
                write!(f, "callback({})", callback_name),
            TaintSource::ConfigValue { name } =>
                write!(f, "config({})", name),
            TaintSource::EntryParameter { function, param_index } =>
                write!(f, "param({}[{}])", function, param_index),
        }
    }
}

// ---------------------------------------------------------------------------
// Taint tags
// ---------------------------------------------------------------------------

/// A taint tag: a source together with the propagation path taken.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TaintTag {
    pub source: TaintSource,
    /// Sequence of instruction IDs through which this taint propagated.
    pub propagation_path: Vec<InstructionId>,
    /// Generation counter — incremented on each propagation step for ordering.
    pub generation: u32,
}

impl TaintTag {
    pub fn new(source: TaintSource) -> Self {
        Self { source, propagation_path: Vec::new(), generation: 0 }
    }

    /// Create a derived tag with one more step in the propagation path.
    pub fn propagate(&self, through: InstructionId) -> Self {
        let mut path = self.propagation_path.clone();
        path.push(through);
        Self {
            source: self.source.clone(),
            propagation_path: path,
            generation: self.generation + 1,
        }
    }

    /// Length of the propagation chain.
    pub fn depth(&self) -> usize {
        self.propagation_path.len()
    }
}

// ---------------------------------------------------------------------------
// Taint lattice
// ---------------------------------------------------------------------------

/// The taint lattice element for a single variable or memory location.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaintValue {
    /// Set of taint tags currently held.
    pub tags: BTreeSet<String>,
    /// Full tag objects (keyed by serialized source for dedup).
    tag_details: HashMap<String, TaintTag>,
}

impl TaintValue {
    pub fn clean() -> Self {
        Self { tags: BTreeSet::new(), tag_details: HashMap::new() }
    }

    pub fn from_tag(tag: TaintTag) -> Self {
        let key = format!("{}", tag.source);
        let mut tv = Self::clean();
        tv.tags.insert(key.clone());
        tv.tag_details.insert(key, tag);
        tv
    }

    pub fn is_tainted(&self) -> bool {
        !self.tags.is_empty()
    }

    /// Join (union) two taint values.
    pub fn join(&self, other: &TaintValue) -> TaintValue {
        let mut result = self.clone();
        for (k, v) in &other.tag_details {
            if !result.tag_details.contains_key(k) || v.generation < result.tag_details[k].generation {
                result.tags.insert(k.clone());
                result.tag_details.insert(k.clone(), v.clone());
            }
        }
        result
    }

    /// Whether this is a superset of other (for fixed-point check).
    pub fn subsumes(&self, other: &TaintValue) -> bool {
        other.tags.is_subset(&self.tags)
    }

    /// Get all taint tags.
    pub fn get_tags(&self) -> Vec<&TaintTag> {
        self.tag_details.values().collect()
    }

    /// Add a tag.
    pub fn add_tag(&mut self, tag: TaintTag) {
        let key = format!("{}", tag.source);
        self.tags.insert(key.clone());
        self.tag_details.insert(key, tag);
    }

    /// Number of distinct taint sources.
    pub fn source_count(&self) -> usize {
        self.tags.len()
    }
}

// ---------------------------------------------------------------------------
// Taint state
// ---------------------------------------------------------------------------

/// Taint state: maps variables and memory locations to their taint values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaintState {
    /// Register-level taints (keyed by register name).
    pub registers: HashMap<String, TaintValue>,
    /// Memory-level taints (keyed by abstract location string).
    pub memory: HashMap<String, TaintValue>,
    /// Function return taints.
    pub return_taint: Option<TaintValue>,
}

impl TaintState {
    pub fn new() -> Self {
        Self {
            registers: HashMap::new(),
            memory: HashMap::new(),
            return_taint: None,
        }
    }

    /// Get taint for a register.
    pub fn get_register(&self, name: &str) -> TaintValue {
        self.registers.get(name).cloned().unwrap_or_else(TaintValue::clean)
    }

    /// Set taint for a register.
    pub fn set_register(&mut self, name: impl Into<String>, taint: TaintValue) {
        let name = name.into();
        if taint.is_tainted() {
            self.registers.insert(name, taint);
        } else {
            self.registers.remove(&name);
        }
    }

    /// Get taint for a memory location.
    pub fn get_memory(&self, loc: &str) -> TaintValue {
        self.memory.get(loc).cloned().unwrap_or_else(TaintValue::clean)
    }

    /// Set taint for a memory location.
    pub fn set_memory(&mut self, loc: impl Into<String>, taint: TaintValue) {
        let loc = loc.into();
        if taint.is_tainted() {
            self.memory.insert(loc, taint);
        } else {
            self.memory.remove(&loc);
        }
    }

    /// Get taint for a Value (register lookup, constant = clean).
    pub fn get_value_taint(&self, val: &Value) -> TaintValue {
        match val {
            Value::Register(name, _) => self.get_register(name),
            Value::GlobalRef(name, _) => self.get_memory(name),
            _ => TaintValue::clean(),
        }
    }

    /// Join two states (pointwise union).
    pub fn join(&self, other: &TaintState) -> TaintState {
        let mut result = self.clone();
        for (k, v) in &other.registers {
            let existing = result.registers.entry(k.clone()).or_insert_with(TaintValue::clean);
            *existing = existing.join(v);
        }
        for (k, v) in &other.memory {
            let existing = result.memory.entry(k.clone()).or_insert_with(TaintValue::clean);
            *existing = existing.join(v);
        }
        if let Some(ref rt) = other.return_taint {
            result.return_taint = Some(match &result.return_taint {
                Some(existing) => existing.join(rt),
                None => rt.clone(),
            });
        }
        result
    }

    /// Check if this state subsumes another (for fixed-point).
    pub fn subsumes(&self, other: &TaintState) -> bool {
        for (k, v) in &other.registers {
            let mine = self.registers.get(k).map(|x| x.clone()).unwrap_or_else(TaintValue::clean);
            if !mine.subsumes(v) { return false; }
        }
        for (k, v) in &other.memory {
            let mine = self.memory.get(k).map(|x| x.clone()).unwrap_or_else(TaintValue::clean);
            if !mine.subsumes(v) { return false; }
        }
        true
    }

    /// Total number of tainted locations.
    pub fn tainted_count(&self) -> usize {
        self.registers.values().filter(|v| v.is_tainted()).count()
            + self.memory.values().filter(|v| v.is_tainted()).count()
    }
}

// ---------------------------------------------------------------------------
// Taint Analysis (intraprocedural)
// ---------------------------------------------------------------------------

/// Forward taint analysis within a single function.
pub struct TaintAnalysis {
    /// Initial taint sources to seed the analysis.
    pub sources: Vec<(String, TaintTag)>,
    /// Maximum iterations for fixed-point.
    pub max_iterations: usize,
    /// Whether to propagate through memory.
    pub track_memory: bool,
    /// Known protocol function patterns for automatic tainting.
    pub protocol_patterns: Vec<String>,
}

impl TaintAnalysis {
    pub fn new() -> Self {
        Self {
            sources: Vec::new(),
            max_iterations: 500,
            track_memory: true,
            protocol_patterns: vec![
                "ssl_cipher".into(), "SSL_CIPHER".into(),
                "ssl_version".into(), "tls_version".into(),
                "ssl_method".into(), "SSL_METHOD".into(),
            ],
        }
    }

    /// Seed a register with a taint tag.
    pub fn add_source(&mut self, register: impl Into<String>, tag: TaintTag) {
        self.sources.push((register.into(), tag));
    }

    /// Automatically discover taint sources in a function based on protocol patterns.
    pub fn discover_sources(&mut self, func: &Function) {
        // Parameters of negotiation functions are taint sources.
        if func.is_negotiation_relevant() {
            for (i, (name, _ty)) in func.params.iter().enumerate() {
                let tag = TaintTag::new(TaintSource::EntryParameter {
                    function: func.name.clone(),
                    param_index: i,
                });
                self.sources.push((name.clone(), tag));
            }
        }

        // Return values of calls to known selection functions.
        for (_bname, _idx, instr) in func.instructions() {
            if let Some(callee) = instr.called_function_name() {
                let callee_lower = callee.to_lowercase();
                if callee_lower.contains("choose") || callee_lower.contains("select")
                    || callee_lower.contains("negotiate")
                {
                    if let Some(dest) = instr.dest() {
                        let tag = TaintTag::new(TaintSource::SelectionFunction {
                            function_name: callee.to_string(),
                        });
                        self.sources.push((dest.to_string(), tag));
                    }
                }
            }
        }
    }

    /// Run forward taint analysis on a function. Returns per-block taint states.
    pub fn analyze_forward(&self, func: &Function) -> SlicerResult<HashMap<String, TaintState>> {
        let block_names: Vec<String> = func.blocks.keys().cloned().collect();
        if block_names.is_empty() {
            return Ok(HashMap::new());
        }

        // Initialize taint state at function entry.
        let mut block_in: HashMap<String, TaintState> = HashMap::new();
        let mut block_out: HashMap<String, TaintState> = HashMap::new();
        for name in &block_names {
            block_in.insert(name.clone(), TaintState::new());
            block_out.insert(name.clone(), TaintState::new());
        }

        // Seed initial taints.
        if let Some(entry_name) = block_names.first() {
            let entry_state = block_in.get_mut(entry_name).unwrap();
            for (reg, tag) in &self.sources {
                entry_state.set_register(reg.clone(), TaintValue::from_tag(tag.clone()));
            }
        }

        // Worklist-based forward iteration.
        let mut worklist: VecDeque<String> = VecDeque::new();
        for name in &block_names {
            worklist.push_back(name.clone());
        }

        let mut iterations = 0;
        while let Some(block_name) = worklist.pop_front() {
            iterations += 1;
            if iterations > self.max_iterations {
                return Err(SlicerError::TaintError(format!(
                    "Forward taint analysis did not converge in {} iterations", self.max_iterations
                )));
            }

            let block = match func.blocks.get(&block_name) {
                Some(b) => b,
                None => continue,
            };

            let in_state = block_in.get(&block_name).unwrap().clone();
            let new_out = self.transfer_block(block, &in_state, &func.name);

            let old_out = block_out.get(&block_name).unwrap();
            if !new_out.subsumes(old_out) || !old_out.subsumes(&new_out) {
                block_out.insert(block_name.clone(), new_out.clone());

                // Propagate to successors.
                for succ in &block.successors {
                    let succ_in = block_in.get(succ).unwrap().clone();
                    let joined = succ_in.join(&new_out);
                    if !succ_in.subsumes(&joined) {
                        block_in.insert(succ.clone(), joined);
                        if !worklist.contains(succ) {
                            worklist.push_back(succ.clone());
                        }
                    }
                }
            }
        }

        log::debug!("Forward taint analysis converged in {} iterations", iterations);
        Ok(block_out)
    }

    /// Run backward taint analysis from sinks to find which sources reach them.
    pub fn analyze_backward(
        &self,
        func: &Function,
        sinks: &[InstructionId],
    ) -> SlicerResult<HashMap<String, TaintState>> {
        let block_names: Vec<String> = func.blocks.keys().cloned().collect();
        if block_names.is_empty() {
            return Ok(HashMap::new());
        }

        let mut block_in: HashMap<String, TaintState> = HashMap::new();
        let mut block_out: HashMap<String, TaintState> = HashMap::new();
        for name in &block_names {
            block_in.insert(name.clone(), TaintState::new());
            block_out.insert(name.clone(), TaintState::new());
        }

        // Seed sinks: mark the values used at each sink instruction as tainted.
        for sink_id in sinks {
            if sink_id.function != func.name { continue; }
            if let Some(block) = func.blocks.get(&sink_id.block) {
                if let Some(instr) = block.instructions.get(sink_id.index) {
                    let state = block_out.get_mut(&sink_id.block).unwrap();
                    for reg in instr.used_registers() {
                        let tag = TaintTag::new(TaintSource::NegotiationField {
                            struct_name: format!("sink@{}", sink_id),
                            field_index: 0,
                        });
                        state.set_register(reg.to_string(), TaintValue::from_tag(tag));
                    }
                }
            }
        }

        // Backward worklist iteration.
        let mut worklist: VecDeque<String> = block_names.iter().rev().cloned().collect();
        let mut iterations = 0;

        while let Some(block_name) = worklist.pop_front() {
            iterations += 1;
            if iterations > self.max_iterations {
                return Err(SlicerError::TaintError(
                    "Backward taint analysis did not converge".into()
                ));
            }

            let block = match func.blocks.get(&block_name) {
                Some(b) => b,
                None => continue,
            };

            let out_state = block_out.get(&block_name).unwrap().clone();
            let new_in = self.transfer_block_backward(block, &out_state, &func.name);

            let old_in = block_in.get(&block_name).unwrap();
            if !new_in.subsumes(old_in) || !old_in.subsumes(&new_in) {
                block_in.insert(block_name.clone(), new_in.clone());

                for pred in &block.predecessors {
                    let pred_out = block_out.get(pred).unwrap().clone();
                    let joined = pred_out.join(&new_in);
                    if !pred_out.subsumes(&joined) {
                        block_out.insert(pred.clone(), joined);
                        if !worklist.contains(pred) {
                            worklist.push_back(pred.clone());
                        }
                    }
                }
            }
        }

        log::debug!("Backward taint analysis converged in {} iterations", iterations);
        Ok(block_in)
    }

    /// Transfer function for a whole basic block (forward).
    fn transfer_block(
        &self,
        block: &BasicBlock,
        in_state: &TaintState,
        func_name: &str,
    ) -> TaintState {
        let mut state = in_state.clone();
        for (idx, instr) in block.instructions.iter().enumerate() {
            let iid = InstructionId::new(func_name, &block.name, idx);
            self.transfer_instruction(instr, &mut state, &iid);
        }
        state
    }

    /// Transfer function for a single instruction (forward taint propagation).
    fn transfer_instruction(
        &self,
        instr: &Instruction,
        state: &mut TaintState,
        iid: &InstructionId,
    ) {
        match instr {
            Instruction::Load { dest, ptr, .. } => {
                // Load propagates taint from the pointer target.
                let ptr_taint = state.get_value_taint(ptr);
                let mem_taint = if let Some(name) = ptr.name() {
                    state.get_memory(name)
                } else {
                    TaintValue::clean()
                };
                let combined = ptr_taint.join(&mem_taint);
                if combined.is_tainted() {
                    let propagated = self.propagate_taint_value(&combined, iid);
                    state.set_register(dest.clone(), propagated);
                }
            }
            Instruction::Store { value, ptr, .. } => {
                let val_taint = state.get_value_taint(value);
                if self.track_memory {
                    if let Some(name) = ptr.name() {
                        if val_taint.is_tainted() {
                            let propagated = self.propagate_taint_value(&val_taint, iid);
                            let existing = state.get_memory(name);
                            state.set_memory(name.to_string(), existing.join(&propagated));
                        }
                    }
                }
            }
            Instruction::BinaryOp { dest, lhs, rhs, .. } => {
                let lt = state.get_value_taint(lhs);
                let rt = state.get_value_taint(rhs);
                let combined = lt.join(&rt);
                if combined.is_tainted() {
                    state.set_register(dest.clone(), self.propagate_taint_value(&combined, iid));
                }
            }
            Instruction::ICmp { dest, lhs, rhs, .. } | Instruction::FCmp { dest, lhs, rhs, .. } => {
                let lt = state.get_value_taint(lhs);
                let rt = state.get_value_taint(rhs);
                let combined = lt.join(&rt);
                if combined.is_tainted() {
                    state.set_register(dest.clone(), self.propagate_taint_value(&combined, iid));
                }
            }
            Instruction::Cast { dest, value, .. } => {
                let t = state.get_value_taint(value);
                if t.is_tainted() {
                    state.set_register(dest.clone(), self.propagate_taint_value(&t, iid));
                }
            }
            Instruction::GetElementPtr { dest, ptr, indices, .. } => {
                let ptr_taint = state.get_value_taint(ptr);
                let idx_taint: TaintValue = indices.iter()
                    .fold(TaintValue::clean(), |acc, idx| acc.join(&state.get_value_taint(idx)));
                let combined = ptr_taint.join(&idx_taint);
                if combined.is_tainted() {
                    state.set_register(dest.clone(), self.propagate_taint_value(&combined, iid));
                }
            }
            Instruction::Phi { dest, incoming, .. } => {
                let mut combined = TaintValue::clean();
                for (val, _bb) in incoming {
                    combined = combined.join(&state.get_value_taint(val));
                }
                if combined.is_tainted() {
                    state.set_register(dest.clone(), self.propagate_taint_value(&combined, iid));
                }
            }
            Instruction::Select { dest, cond, true_val, false_val, .. } => {
                let ct = state.get_value_taint(cond);
                let tt = state.get_value_taint(true_val);
                let ft = state.get_value_taint(false_val);
                let combined = ct.join(&tt).join(&ft);
                if combined.is_tainted() {
                    state.set_register(dest.clone(), self.propagate_taint_value(&combined, iid));
                }
            }
            Instruction::Call { dest, func, args, .. } => {
                // Union taint from all arguments.
                let mut arg_taint = TaintValue::clean();
                for arg in args {
                    arg_taint = arg_taint.join(&state.get_value_taint(arg));
                }
                // Also check if the callee itself is tainted (indirect call through tainted ptr).
                let func_taint = state.get_value_taint(func);
                let combined = arg_taint.join(&func_taint);

                // Check if callee is a known protocol function that introduces taint.
                if let Some(callee_name) = match func {
                    Value::FunctionRef(n, _) | Value::GlobalRef(n, _) => Some(n.as_str()),
                    _ => None,
                } {
                    let callee_lower = callee_name.to_lowercase();
                    let is_selection = callee_lower.contains("choose")
                        || callee_lower.contains("select")
                        || callee_lower.contains("negotiate");
                    if is_selection {
                        if let Some(d) = dest {
                            let mut new_taint = combined.clone();
                            new_taint.add_tag(TaintTag::new(TaintSource::SelectionFunction {
                                function_name: callee_name.to_string(),
                            }));
                            state.set_register(d.clone(), self.propagate_taint_value(&new_taint, iid));
                            return;
                        }
                    }
                }

                if let Some(d) = dest {
                    if combined.is_tainted() {
                        state.set_register(d.clone(), self.propagate_taint_value(&combined, iid));
                    }
                }

                // Tainted arguments may taint memory through the callee.
                if self.track_memory && combined.is_tainted() {
                    for arg in args {
                        if let Some(name) = arg.name() {
                            if arg.ty().is_pointer() {
                                let existing = state.get_memory(name);
                                state.set_memory(name.to_string(),
                                    existing.join(&self.propagate_taint_value(&combined, iid)));
                            }
                        }
                    }
                }
            }
            Instruction::ExtractValue { dest, aggregate, .. } => {
                let t = state.get_value_taint(aggregate);
                if t.is_tainted() {
                    state.set_register(dest.clone(), self.propagate_taint_value(&t, iid));
                }
            }
            Instruction::InsertValue { dest, aggregate, value, .. } => {
                let at = state.get_value_taint(aggregate);
                let vt = state.get_value_taint(value);
                let combined = at.join(&vt);
                if combined.is_tainted() {
                    state.set_register(dest.clone(), self.propagate_taint_value(&combined, iid));
                }
            }
            Instruction::Freeze { dest, value, .. } => {
                let t = state.get_value_taint(value);
                if t.is_tainted() {
                    state.set_register(dest.clone(), self.propagate_taint_value(&t, iid));
                }
            }
            Instruction::Alloca { dest, .. } => {
                // Alloca itself is not tainted, but the allocated location is trackable.
                state.set_register(dest.clone(), TaintValue::clean());
            }
            Instruction::Ret { value } => {
                if let Some(v) = value {
                    let t = state.get_value_taint(v);
                    if t.is_tainted() {
                        state.return_taint = Some(self.propagate_taint_value(&t, iid));
                    }
                }
            }
            Instruction::MemIntrinsic { dest_ptr, src_or_val, .. } => {
                let src_taint = state.get_value_taint(src_or_val);
                if self.track_memory && src_taint.is_tainted() {
                    if let Some(name) = dest_ptr.name() {
                        let existing = state.get_memory(name);
                        state.set_memory(name.to_string(),
                            existing.join(&self.propagate_taint_value(&src_taint, iid)));
                    }
                }
            }
            _ => {}
        }
    }

    /// Transfer function for backward analysis (one block).
    fn transfer_block_backward(
        &self,
        block: &BasicBlock,
        out_state: &TaintState,
        func_name: &str,
    ) -> TaintState {
        let mut state = out_state.clone();
        for (idx, instr) in block.instructions.iter().enumerate().rev() {
            let iid = InstructionId::new(func_name, &block.name, idx);
            self.transfer_instruction_backward(instr, &mut state, &iid);
        }
        state
    }

    /// Backward transfer: if the destination is tainted, taint the operands.
    fn transfer_instruction_backward(
        &self,
        instr: &Instruction,
        state: &mut TaintState,
        iid: &InstructionId,
    ) {
        if let Some(dest) = instr.dest() {
            let dest_taint = state.get_register(dest);
            if dest_taint.is_tainted() {
                let propagated = self.propagate_taint_value(&dest_taint, iid);
                for reg in instr.used_registers() {
                    let existing = state.get_register(reg);
                    state.set_register(reg.to_string(), existing.join(&propagated));
                }
            }
        }
        // For stores: if the memory location is tainted, taint the stored value.
        if let Instruction::Store { value, ptr, .. } = instr {
            if let Some(name) = ptr.name() {
                let mem_taint = state.get_memory(name);
                if mem_taint.is_tainted() {
                    let propagated = self.propagate_taint_value(&mem_taint, iid);
                    if let Some(vname) = value.name() {
                        let existing = state.get_register(vname);
                        state.set_register(vname.to_string(), existing.join(&propagated));
                    }
                }
            }
        }
    }

    /// Propagate all tags through an instruction ID.
    fn propagate_taint_value(&self, tv: &TaintValue, iid: &InstructionId) -> TaintValue {
        let mut result = TaintValue::clean();
        for tag in tv.get_tags() {
            result.add_tag(tag.propagate(iid.clone()));
        }
        result
    }
}

// ---------------------------------------------------------------------------
// Interprocedural Taint Analysis
// ---------------------------------------------------------------------------

/// Summary of taint effects for a function.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionTaintSummary {
    pub function_name: String,
    /// Which parameters, when tainted, taint the return value.
    pub param_to_return: Vec<usize>,
    /// Which parameters, when tainted, taint other parameters (via pointer writes).
    pub param_to_param: Vec<(usize, usize)>,
    /// Whether the function introduces new taint (is a taint source).
    pub is_source: bool,
    /// The sources this function introduces.
    pub introduced_sources: Vec<TaintSource>,
}

/// Interprocedural taint analysis using function summaries.
pub struct InterprocTaintAnalysis {
    /// Per-function summaries.
    pub summaries: HashMap<String, FunctionTaintSummary>,
    /// Configuration.
    pub max_iterations: usize,
    pub max_call_depth: usize,
    pub track_memory: bool,
}

impl InterprocTaintAnalysis {
    pub fn new() -> Self {
        Self {
            summaries: HashMap::new(),
            max_iterations: 1000,
            max_call_depth: 10,
            track_memory: true,
        }
    }

    /// Build summaries for all functions in the module.
    pub fn build_summaries(&mut self, module: &Module) -> SlicerResult<()> {
        // First pass: identify functions that are taint sources.
        for (name, func) in &module.functions {
            let mut summary = FunctionTaintSummary {
                function_name: name.clone(),
                param_to_return: Vec::new(),
                param_to_param: Vec::new(),
                is_source: false,
                introduced_sources: Vec::new(),
            };

            let name_lower = name.to_lowercase();
            if name_lower.contains("choose") || name_lower.contains("select")
                || name_lower.contains("negotiate")
            {
                summary.is_source = true;
                summary.introduced_sources.push(TaintSource::SelectionFunction {
                    function_name: name.clone(),
                });
            }

            // Analyze which params flow to the return value.
            if !func.is_declaration {
                for (i, (pname, _)) in func.params.iter().enumerate() {
                    let mut analysis = TaintAnalysis::new();
                    analysis.max_iterations = self.max_iterations;
                    analysis.track_memory = self.track_memory;
                    analysis.add_source(pname.clone(), TaintTag::new(
                        TaintSource::EntryParameter {
                            function: name.clone(),
                            param_index: i,
                        }
                    ));

                    if let Ok(results) = analysis.analyze_forward(func) {
                        // Check if any return instruction has tainted value.
                        for (_bname, state) in &results {
                            if let Some(ref rt) = state.return_taint {
                                if rt.is_tainted() {
                                    summary.param_to_return.push(i);
                                    break;
                                }
                            }
                        }

                        // Check param-to-param flow (pointer args).
                        for (j, (other_name, other_ty)) in func.params.iter().enumerate() {
                            if i == j { continue; }
                            if other_ty.is_pointer() {
                                for (_bname, state) in &results {
                                    let mem_taint = state.get_memory(other_name);
                                    if mem_taint.is_tainted() {
                                        summary.param_to_param.push((i, j));
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            self.summaries.insert(name.clone(), summary);
        }

        Ok(())
    }

    /// Apply summaries to resolve taint through function calls.
    pub fn resolve_call_taint(
        &self,
        callee_name: &str,
        arg_taints: &[TaintValue],
        iid: &InstructionId,
    ) -> (TaintValue, Vec<(usize, TaintValue)>) {
        let mut ret_taint = TaintValue::clean();
        let mut param_effects: Vec<(usize, TaintValue)> = Vec::new();

        if let Some(summary) = self.summaries.get(callee_name) {
            // Propagate taint from args to return.
            for &param_idx in &summary.param_to_return {
                if param_idx < arg_taints.len() && arg_taints[param_idx].is_tainted() {
                    for tag in arg_taints[param_idx].get_tags() {
                        ret_taint.add_tag(tag.propagate(iid.clone()));
                    }
                }
            }

            // Propagate taint between parameters.
            for &(src_idx, dst_idx) in &summary.param_to_param {
                if src_idx < arg_taints.len() && arg_taints[src_idx].is_tainted() {
                    let mut effect = TaintValue::clean();
                    for tag in arg_taints[src_idx].get_tags() {
                        effect.add_tag(tag.propagate(iid.clone()));
                    }
                    param_effects.push((dst_idx, effect));
                }
            }

            // Handle functions that are themselves sources.
            if summary.is_source {
                for source in &summary.introduced_sources {
                    ret_taint.add_tag(TaintTag::new(source.clone()));
                }
            }
        } else {
            // Unknown function: conservatively taint return from all tainted args.
            for arg_taint in arg_taints {
                if arg_taint.is_tainted() {
                    for tag in arg_taint.get_tags() {
                        ret_taint.add_tag(tag.propagate(iid.clone()));
                    }
                }
            }
        }

        (ret_taint, param_effects)
    }

    /// Get the summary for a function.
    pub fn summary(&self, func_name: &str) -> Option<&FunctionTaintSummary> {
        self.summaries.get(func_name)
    }

    /// Get all functions that are taint sources.
    pub fn source_functions(&self) -> Vec<&str> {
        self.summaries.iter()
            .filter(|(_, s)| s.is_source)
            .map(|(name, _)| name.as_str())
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Taint query helpers
// ---------------------------------------------------------------------------

/// Query result: which instructions are tainted and by what sources.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaintQueryResult {
    pub tainted_instructions: Vec<TaintedInstruction>,
    pub total_instructions: usize,
    pub taint_coverage: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaintedInstruction {
    pub id: InstructionId,
    pub sources: Vec<String>,
}

/// Run a complete taint query on a function.
pub fn query_tainted_instructions(
    func: &Function,
    sources: Vec<(String, TaintTag)>,
) -> SlicerResult<TaintQueryResult> {
    let mut analysis = TaintAnalysis::new();
    for (reg, tag) in sources {
        analysis.add_source(reg, tag);
    }

    let results = analysis.analyze_forward(func)?;

    let mut tainted = Vec::new();
    let mut total = 0usize;

    for (bname, block) in &func.blocks {
        for (idx, instr) in block.instructions.iter().enumerate() {
            total += 1;
            if let Some(state) = results.get(bname) {
                // Check if any operand is tainted.
                let is_tainted = instr.used_registers().iter().any(|reg| {
                    state.get_register(reg).is_tainted()
                });
                if is_tainted {
                    let sources: Vec<String> = instr.used_registers().iter()
                        .flat_map(|reg| {
                            state.get_register(reg).tags.iter().cloned().collect::<Vec<_>>()
                        })
                        .collect::<HashSet<_>>()
                        .into_iter()
                        .collect();
                    tainted.push(TaintedInstruction {
                        id: InstructionId::new(&func.name, bname, idx),
                        sources,
                    });
                }
            }
        }
    }

    let coverage = if total > 0 { tainted.len() as f64 / total as f64 } else { 0.0 };
    Ok(TaintQueryResult {
        tainted_instructions: tainted,
        total_instructions: total,
        taint_coverage: coverage,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Module, Function, BasicBlock, Instruction, Value, Type};

    fn make_test_function() -> Function {
        let mut func = Function::new("test_cipher_select", Type::i32());
        func.add_param("ctx", Type::ptr(Type::NamedStruct("SSL_CTX".into())));
        func.add_param("ciphers", Type::ptr(Type::NamedStruct("STACK".into())));

        let entry = func.add_block("entry");
        entry.push(Instruction::Load {
            dest: "cipher_list".into(),
            ty: Type::ptr(Type::i8()),
            ptr: Value::reg("ciphers", Type::ptr(Type::NamedStruct("STACK".into()))),
            volatile: false,
            align: Some(8),
        });
        entry.push(Instruction::Call {
            dest: Some("chosen".into()),
            func: Value::func_ref("ssl3_choose_cipher", Type::func(Type::i32(), vec![])),
            args: vec![
                Value::reg("ctx", Type::ptr(Type::NamedStruct("SSL_CTX".into()))),
                Value::reg("cipher_list", Type::ptr(Type::i8())),
            ],
            ret_ty: Type::i32(),
            is_tail: false,
            calling_conv: None,
            attrs: vec![],
        });
        entry.push(Instruction::ICmp {
            dest: "is_null".into(),
            pred: crate::ir::IntPredicate::Eq,
            lhs: Value::reg("chosen", Type::i32()),
            rhs: Value::int(0, 32),
        });
        entry.push(Instruction::CondBr {
            cond: Value::reg("is_null", Type::Int(1)),
            true_dest: "error".into(),
            false_dest: "ok".into(),
        });

        let error = func.add_block("error");
        error.push(Instruction::Ret { value: Some(Value::int(-1, 32)) });

        let ok = func.add_block("ok");
        ok.push(Instruction::Ret { value: Some(Value::reg("chosen", Type::i32())) });

        func.compute_predecessors();
        func
    }

    #[test]
    fn test_taint_source_display() {
        let src = TaintSource::CipherSuiteList { is_client: true };
        assert_eq!(format!("{}", src), "cipher_list(client)");
    }

    #[test]
    fn test_taint_tag_propagation() {
        let tag = TaintTag::new(TaintSource::VersionField { field: "major".into() });
        assert_eq!(tag.depth(), 0);
        let derived = tag.propagate(InstructionId::new("f", "bb", 0));
        assert_eq!(derived.depth(), 1);
        assert_eq!(derived.generation, 1);
    }

    #[test]
    fn test_taint_value_join() {
        let a = TaintValue::from_tag(TaintTag::new(TaintSource::VersionField { field: "major".into() }));
        let b = TaintValue::from_tag(TaintTag::new(TaintSource::CipherSuiteList { is_client: true }));
        let joined = a.join(&b);
        assert_eq!(joined.source_count(), 2);
        assert!(joined.is_tainted());
    }

    #[test]
    fn test_taint_state_operations() {
        let mut state = TaintState::new();
        assert_eq!(state.tainted_count(), 0);

        state.set_register("x", TaintValue::from_tag(
            TaintTag::new(TaintSource::VersionField { field: "minor".into() })
        ));
        assert_eq!(state.tainted_count(), 1);
        assert!(state.get_register("x").is_tainted());
        assert!(!state.get_register("y").is_tainted());
    }

    #[test]
    fn test_taint_state_join() {
        let mut s1 = TaintState::new();
        s1.set_register("a", TaintValue::from_tag(
            TaintTag::new(TaintSource::VersionField { field: "v".into() })
        ));
        let mut s2 = TaintState::new();
        s2.set_register("b", TaintValue::from_tag(
            TaintTag::new(TaintSource::CipherSuiteList { is_client: false })
        ));
        let joined = s1.join(&s2);
        assert!(joined.get_register("a").is_tainted());
        assert!(joined.get_register("b").is_tainted());
    }

    #[test]
    fn test_forward_taint_analysis() {
        let func = make_test_function();
        let mut analysis = TaintAnalysis::new();
        analysis.add_source("ciphers", TaintTag::new(
            TaintSource::CipherSuiteList { is_client: true }
        ));

        let result = analysis.analyze_forward(&func).unwrap();
        assert!(!result.is_empty());

        // The cipher_list register should be tainted (loaded from tainted ciphers).
        if let Some(entry_state) = result.get("entry") {
            assert!(entry_state.get_register("cipher_list").is_tainted()
                || entry_state.get_register("chosen").is_tainted());
        }
    }

    #[test]
    fn test_backward_taint_analysis() {
        let func = make_test_function();
        let analysis = TaintAnalysis::new();
        let sinks = vec![
            InstructionId::new("test_cipher_select", "ok", 0),
        ];
        let result = analysis.analyze_backward(&func, &sinks).unwrap();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_discover_sources() {
        let func = make_test_function();
        let mut analysis = TaintAnalysis::new();
        analysis.discover_sources(&func);
        assert!(!analysis.sources.is_empty());
    }

    #[test]
    fn test_taint_source_classification() {
        let cipher = TaintSource::CipherSuiteList { is_client: true };
        assert!(cipher.is_cipher_related());
        assert!(!cipher.is_version_related());

        let version = TaintSource::VersionField { field: "major".into() };
        assert!(version.is_version_related());
        assert!(!version.is_cipher_related());
    }

    #[test]
    fn test_interproc_analysis_summaries() {
        let module = Module::test_module();
        let mut analysis = InterprocTaintAnalysis::new();
        analysis.build_summaries(&module).unwrap();
        assert!(!analysis.summaries.is_empty());
    }

    #[test]
    fn test_interproc_resolve_unknown() {
        let analysis = InterprocTaintAnalysis::new();
        let arg_taints = vec![
            TaintValue::from_tag(TaintTag::new(TaintSource::VersionField { field: "x".into() })),
        ];
        let iid = InstructionId::new("f", "bb", 0);
        let (ret, effects) = analysis.resolve_call_taint("unknown_func", &arg_taints, &iid);
        assert!(ret.is_tainted()); // conservative
        assert!(effects.is_empty());
    }

    #[test]
    fn test_query_tainted_instructions() {
        let func = make_test_function();
        let sources = vec![
            ("ciphers".to_string(), TaintTag::new(TaintSource::CipherSuiteList { is_client: true })),
        ];
        let result = query_tainted_instructions(&func, sources).unwrap();
        assert!(result.total_instructions > 0);
    }

    #[test]
    fn test_taint_value_clean() {
        let tv = TaintValue::clean();
        assert!(!tv.is_tainted());
        assert_eq!(tv.source_count(), 0);
    }

    #[test]
    fn test_taint_value_subsumes() {
        let a = TaintValue::from_tag(TaintTag::new(TaintSource::VersionField { field: "a".into() }));
        let b = TaintValue::clean();
        assert!(a.subsumes(&b));
        assert!(!b.subsumes(&a));
    }
}
