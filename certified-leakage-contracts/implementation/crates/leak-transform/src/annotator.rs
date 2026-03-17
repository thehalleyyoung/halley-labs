//! Security annotation for the analysis IR.
//!
//! Provides mechanisms for marking IR instructions and memory regions with
//! security levels (public/secret), enabling downstream leakage analysis to
//! determine which operations may depend on secret data.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use shared_types::{AddressRange, RegisterId, SecurityLevel, VirtualAddress};

use crate::ir::{AnalysisIR, IRInstruction};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// How annotations are sourced.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AnnotationSource {
    /// Annotations provided by the user via a configuration file.
    UserConfig,
    /// Annotations inferred from symbol / DWARF debug info.
    DebugInfo,
    /// Annotations inferred by inter-procedural taint analysis.
    TaintAnalysis,
    /// Annotations derived from known cryptographic API patterns.
    CryptoPatternMatch,
    /// Manually specified inline annotations (e.g. `__attribute__`).
    InlineAttribute,
}

/// Configuration controlling how annotations are applied.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnotationConfig {
    /// Sources from which to gather annotations (tried in order).
    pub sources: Vec<AnnotationSource>,
    /// If `true`, conservatively treat unknown memory as secret.
    pub default_secret: bool,
    /// Patterns of function names whose inputs should be marked secret.
    pub secret_function_patterns: Vec<String>,
    /// Explicit register→security-level overrides at function entry.
    pub entry_register_levels: HashMap<RegisterId, SecurityLevel>,
}

impl Default for AnnotationConfig {
    fn default() -> Self {
        Self {
            sources: vec![AnnotationSource::UserConfig, AnnotationSource::TaintAnalysis],
            default_secret: false,
            secret_function_patterns: vec![
                "aes_".to_string(),
                "chacha".to_string(),
                "poly1305".to_string(),
            ],
            entry_register_levels: HashMap::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// SecretRegions
// ---------------------------------------------------------------------------

/// Tracks which address ranges and registers are classified as secret.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SecretRegions {
    /// Memory address ranges holding secret data.
    pub memory_ranges: Vec<AddressRange>,
    /// Registers currently holding secret values.
    pub secret_registers: HashSet<RegisterId>,
    /// Stack offsets (relative to RSP at function entry) that are secret.
    pub secret_stack_offsets: HashSet<i64>,
}

impl SecretRegions {
    pub fn new() -> Self {
        Self::default()
    }

    /// Mark an address range as secret.
    pub fn add_memory_range(&mut self, range: AddressRange) {
        self.memory_ranges.push(range);
    }

    /// Mark a register as secret.
    pub fn add_register(&mut self, reg: RegisterId) {
        self.secret_registers.insert(reg);
    }

    /// Mark a stack offset as secret.
    pub fn add_stack_offset(&mut self, offset: i64) {
        self.secret_stack_offsets.insert(offset);
    }

    /// Check whether a given address falls within a secret region.
    pub fn is_secret_address(&self, addr: VirtualAddress) -> bool {
        self.memory_ranges.iter().any(|r| r.contains(addr))
    }

    /// Check whether a register is currently secret.
    pub fn is_secret_register(&self, reg: RegisterId) -> bool {
        self.secret_registers.contains(&reg)
    }

    /// Total number of bytes classified as secret.
    pub fn total_secret_bytes(&self) -> u64 {
        self.memory_ranges.iter().map(|r| r.len()).sum()
    }
}

// ---------------------------------------------------------------------------
// TaintAnnotator
// ---------------------------------------------------------------------------

/// Forward taint propagation engine that walks the IR to compute which
/// instructions operate on secret data.
#[derive(Debug, Clone)]
pub struct TaintAnnotator {
    /// Current taint state: register → security level.
    register_taint: HashMap<RegisterId, SecurityLevel>,
    /// Tainted memory locations (simplified to addresses).
    memory_taint: HashMap<VirtualAddress, SecurityLevel>,
    /// Number of instructions processed.
    instructions_processed: u64,
}

impl TaintAnnotator {
    pub fn new() -> Self {
        Self {
            register_taint: HashMap::new(),
            memory_taint: HashMap::new(),
            instructions_processed: 0,
        }
    }

    /// Seed initial taint from [`SecretRegions`].
    pub fn seed(&mut self, regions: &SecretRegions) {
        for &reg in &regions.secret_registers {
            self.register_taint.insert(reg, SecurityLevel::Secret);
        }
    }

    /// Mark a register as tainted (secret).
    pub fn taint_register(&mut self, reg: RegisterId) {
        self.register_taint.insert(reg, SecurityLevel::Secret);
    }

    /// Query the taint level of a register.
    pub fn register_level(&self, reg: RegisterId) -> SecurityLevel {
        self.register_taint
            .get(&reg)
            .copied()
            .unwrap_or(SecurityLevel::Public)
    }

    /// Propagate taint across a single instruction, returning its computed
    /// security level.
    pub fn propagate_instruction(&mut self, instr: &IRInstruction) -> SecurityLevel {
        self.instructions_processed += 1;

        // Join the security levels of all used registers.
        let mut level = SecurityLevel::Public;
        for u in &instr.uses {
            for reg in u.referenced_registers() {
                level = level.join(self.register_level(reg));
            }
        }

        // Propagate to all defined registers.
        for d in &instr.defs {
            if let crate::ir::IROperand::Register { id, .. } = d {
                self.register_taint.insert(*id, level);
            }
        }

        level
    }

    /// Number of instructions processed so far.
    pub fn instructions_processed(&self) -> u64 {
        self.instructions_processed
    }

    /// Reset all taint state.
    pub fn reset(&mut self) {
        self.register_taint.clear();
        self.memory_taint.clear();
        self.instructions_processed = 0;
    }
}

impl Default for TaintAnnotator {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// AnnotatedIR
// ---------------------------------------------------------------------------

/// The analysis IR augmented with per-instruction security annotations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnotatedIR {
    /// Underlying analysis IR (with `annotated` set to `true`).
    pub ir: AnalysisIR,
    /// Per-address security level (instruction address → level).
    pub instruction_levels: HashMap<VirtualAddress, SecurityLevel>,
    /// Secret regions that were used for annotation.
    pub secret_regions: SecretRegions,
    /// Which annotation sources contributed.
    pub sources_used: Vec<AnnotationSource>,
}

impl AnnotatedIR {
    pub fn new(ir: AnalysisIR, secret_regions: SecretRegions) -> Self {
        Self {
            ir,
            instruction_levels: HashMap::new(),
            secret_regions,
            sources_used: Vec::new(),
        }
    }

    /// Query the security level of the instruction at `addr`.
    pub fn level_at(&self, addr: VirtualAddress) -> SecurityLevel {
        self.instruction_levels
            .get(&addr)
            .copied()
            .unwrap_or(SecurityLevel::Public)
    }

    /// Number of instructions classified as secret.
    pub fn secret_instruction_count(&self) -> usize {
        self.instruction_levels
            .values()
            .filter(|l| l.is_secret())
            .count()
    }

    /// Number of instructions classified as public.
    pub fn public_instruction_count(&self) -> usize {
        self.instruction_levels
            .values()
            .filter(|l| l.is_public())
            .count()
    }
}

// ---------------------------------------------------------------------------
// SecurityAnnotator
// ---------------------------------------------------------------------------

/// Top-level orchestrator that combines annotation sources to produce an
/// [`AnnotatedIR`].
#[derive(Debug, Clone)]
pub struct SecurityAnnotator {
    config: AnnotationConfig,
}

impl SecurityAnnotator {
    pub fn new(config: AnnotationConfig) -> Self {
        Self { config }
    }

    pub fn with_defaults() -> Self {
        Self::new(AnnotationConfig::default())
    }

    /// Annotate the given [`AnalysisIR`] using the configured sources.
    pub fn annotate(
        &self,
        mut ir: AnalysisIR,
        secret_regions: &SecretRegions,
    ) -> AnnotatedIR {
        let mut annotator = TaintAnnotator::new();
        annotator.seed(secret_regions);

        // Seed entry-register overrides.
        for (&reg, &level) in &self.config.entry_register_levels {
            if level.is_secret() {
                annotator.taint_register(reg);
            }
        }

        let mut instruction_levels = HashMap::new();

        // Walk every function / block / instruction.
        for func in ir.program.functions.values_mut() {
            for block in func.blocks.values_mut() {
                for instr in &mut block.instructions {
                    let level = annotator.propagate_instruction(instr);
                    instr.security_level = Some(level);
                    instruction_levels.insert(instr.address, level);
                }
            }
        }

        ir.annotated = true;

        AnnotatedIR {
            ir,
            instruction_levels,
            secret_regions: secret_regions.clone(),
            sources_used: self.config.sources.clone(),
        }
    }

    /// Return a reference to the active configuration.
    pub fn config(&self) -> &AnnotationConfig {
        &self.config
    }
}
